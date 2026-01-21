//! Multi-chunk world rendering for procedural terrain.
//!
//! Manages GPU resources for rendering multiple SVO-DAG chunks simultaneously.

use ash::vk;
use glam::Vec3;
use gpu_allocator::MemoryLocation;
use hashbrown::HashMap;
use voxelicous_core::constants::CHUNK_SIZE;
use voxelicous_core::coords::ChunkPos;
use voxelicous_core::math::{Aabb, Frustum};
use voxelicous_gpu::error::Result;
use voxelicous_gpu::memory::{GpuAllocator, GpuBuffer};
use voxelicous_voxel::{SvoDag, VoxelStorage};

use crate::GpuSvoDag;

/// GPU-side information about a single chunk for shader access.
///
/// This struct is uploaded to a GPU buffer and accessed via SSBO in shaders.
/// Layout must match the shader struct exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuChunkInfo {
    /// Device address of this chunk's node buffer.
    pub node_buffer_address: u64,
    /// Root node index.
    pub root_index: u32,
    /// Octree depth.
    pub octree_depth: u32,
    /// World-space offset X.
    pub world_offset_x: f32,
    /// World-space offset Y.
    pub world_offset_y: f32,
    /// World-space offset Z.
    pub world_offset_z: f32,
    /// Padding for 32-byte alignment.
    pub _padding: f32,
}

impl GpuChunkInfo {
    /// Create chunk info from a GPU DAG and chunk position.
    pub fn from_gpu_dag(gpu_dag: &GpuSvoDag, pos: ChunkPos) -> Self {
        Self {
            node_buffer_address: gpu_dag.device_address,
            root_index: gpu_dag.root_index,
            octree_depth: gpu_dag.depth,
            world_offset_x: (pos.x as f32) * CHUNK_SIZE as f32,
            world_offset_y: (pos.y as f32) * CHUNK_SIZE as f32,
            world_offset_z: (pos.z as f32) * CHUNK_SIZE as f32,
            _padding: 0.0,
        }
    }

    /// Size in bytes.
    pub const SIZE: usize = std::mem::size_of::<Self>();
}

/// Information about a loaded chunk on the GPU.
struct LoadedChunk {
    /// The uploaded GPU DAG.
    gpu_dag: GpuSvoDag,
    /// Chunk info struct for shader access.
    info: GpuChunkInfo,
}

/// Push constants for multi-chunk ray marching.
///
/// This struct is passed to the shader for each frame.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WorldRenderPushConstants {
    /// Render target dimensions (width, height).
    pub screen_size: [u32; 2],
    /// Maximum traversal steps per chunk.
    pub max_steps: u32,
    /// Number of active chunks.
    pub chunk_count: u32,
    /// Device address of the chunk info buffer.
    pub chunk_info_address: u64,
}

impl WorldRenderPushConstants {
    /// Size in bytes.
    pub const SIZE: u32 = std::mem::size_of::<Self>() as u32;
}

/// Manages GPU resources for multi-chunk world rendering.
pub struct WorldRenderer {
    /// Per-chunk GPU data indexed by position.
    chunks: HashMap<ChunkPos, LoadedChunk>,
    /// GPU buffer containing chunk info array for shader access.
    chunk_info_buffer: Option<GpuBuffer>,
    /// Device address of chunk info buffer.
    chunk_info_address: vk::DeviceAddress,
    /// Maximum chunks to keep on GPU.
    #[allow(dead_code)]
    max_gpu_chunks: usize,
    /// Whether the chunk info buffer needs rebuilding.
    dirty: bool,
}

impl WorldRenderer {
    /// Create a new world renderer.
    pub fn new(max_gpu_chunks: usize) -> Self {
        Self {
            chunks: HashMap::with_capacity(max_gpu_chunks),
            chunk_info_buffer: None,
            chunk_info_address: 0,
            max_gpu_chunks,
            dirty: true,
        }
    }

    /// Upload a chunk's DAG to the GPU.
    ///
    /// If a chunk already exists at this position, waits for GPU idle before replacing.
    /// Returns `Ok(true)` if the chunk was uploaded, `Ok(false)` if skipped (empty/air-only).
    pub fn upload_chunk(
        &mut self,
        allocator: &mut GpuAllocator,
        device: &ash::Device,
        pos: ChunkPos,
        dag: &SvoDag,
    ) -> Result<bool> {
        // Skip empty (air-only) chunks - no point uploading them
        if dag.is_empty() {
            // Still remove old chunk if it existed (chunk might have been cleared)
            if let Some(old) = self.chunks.remove(&pos) {
                unsafe {
                    let _ = device.device_wait_idle();
                }
                let mut buffer = old.gpu_dag.node_buffer;
                allocator.free_buffer(&mut buffer)?;
                self.dirty = true;
            }
            return Ok(false);
        }

        // Remove old chunk if it exists (wait for GPU first)
        if let Some(old) = self.chunks.remove(&pos) {
            unsafe {
                let _ = device.device_wait_idle();
            }
            let mut buffer = old.gpu_dag.node_buffer;
            allocator.free_buffer(&mut buffer)?;
        }

        // Upload new chunk
        let gpu_dag = GpuSvoDag::upload(allocator, device, dag)?;
        let info = GpuChunkInfo::from_gpu_dag(&gpu_dag, pos);

        self.chunks.insert(pos, LoadedChunk { gpu_dag, info });
        self.dirty = true;

        Ok(true)
    }

    /// Remove a chunk from GPU memory.
    ///
    /// Waits for GPU idle before freeing to ensure buffer is not in use.
    pub fn remove_chunk(
        &mut self,
        allocator: &mut GpuAllocator,
        device: &ash::Device,
        pos: ChunkPos,
    ) -> Result<()> {
        if let Some(mut loaded) = self.chunks.remove(&pos) {
            unsafe {
                let _ = device.device_wait_idle();
            }
            allocator.free_buffer(&mut loaded.gpu_dag.node_buffer)?;
            self.dirty = true;
        }
        Ok(())
    }

    /// Check if a chunk is loaded on GPU.
    pub fn has_chunk(&self, pos: ChunkPos) -> bool {
        self.chunks.contains_key(&pos)
    }

    /// Get the number of loaded chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Get total GPU memory usage in bytes.
    pub fn gpu_memory_usage(&self) -> u64 {
        let dag_memory: u64 = self.chunks.values().map(|c| c.gpu_dag.buffer_size()).sum();
        let info_memory = self.chunk_info_buffer.as_ref().map_or(0, |b| b.size);
        dag_memory + info_memory
    }

    /// Rebuild the chunk info buffer for shader access.
    ///
    /// Call this before rendering if chunks have changed.
    ///
    /// # Safety
    /// This function waits for the GPU to be idle before freeing the old buffer
    /// to ensure it's not in use. This may impact performance.
    pub fn rebuild_chunk_info_buffer(
        &mut self,
        allocator: &mut GpuAllocator,
        device: &ash::Device,
    ) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }

        // Wait for GPU to finish before freeing old buffer
        // This prevents accessing freed memory from previous frames
        if self.chunk_info_buffer.is_some() {
            unsafe {
                let _ = device.device_wait_idle();
            }
        }

        // Free old buffer (now safe since GPU is idle)
        if let Some(mut old_buffer) = self.chunk_info_buffer.take() {
            allocator.free_buffer(&mut old_buffer)?;
        }

        if self.chunks.is_empty() {
            self.chunk_info_address = 0;
            self.dirty = false;
            return Ok(());
        }

        // Collect chunk infos
        let infos: Vec<GpuChunkInfo> = self.chunks.values().map(|c| c.info).collect();

        // Create new buffer
        let buffer_size = (infos.len() * GpuChunkInfo::SIZE) as u64;
        let usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        let chunk_info_buffer = allocator.create_buffer(
            buffer_size,
            usage,
            MemoryLocation::CpuToGpu,
            "world_chunk_info",
        )?;

        // Write data
        chunk_info_buffer.write::<u8>(bytemuck::cast_slice(&infos))?;

        // Get device address
        self.chunk_info_address = chunk_info_buffer.device_address(device);
        self.chunk_info_buffer = Some(chunk_info_buffer);
        self.dirty = false;

        Ok(())
    }

    /// Rebuild the chunk info buffer with frustum culling and distance sorting.
    ///
    /// This optimized version:
    /// 1. Culls chunks outside the view frustum
    /// 2. Sorts visible chunks front-to-back by distance to camera
    ///
    /// The front-to-back ordering improves early termination in the shader
    /// since closer chunks set `closest.t` first.
    ///
    /// # Safety
    /// This function waits for the GPU to be idle before freeing the old buffer.
    pub fn rebuild_chunk_info_buffer_culled(
        &mut self,
        allocator: &mut GpuAllocator,
        device: &ash::Device,
        frustum: &Frustum,
        camera_pos: Vec3,
    ) -> Result<()> {
        // Always rebuild when using culling since visibility changes with camera
        // Wait for GPU to finish before freeing old buffer
        if self.chunk_info_buffer.is_some() {
            unsafe {
                let _ = device.device_wait_idle();
            }
        }

        // Free old buffer
        if let Some(mut old_buffer) = self.chunk_info_buffer.take() {
            allocator.free_buffer(&mut old_buffer)?;
        }

        if self.chunks.is_empty() {
            self.chunk_info_address = 0;
            self.dirty = false;
            return Ok(());
        }

        // Collect visible chunks with distances
        let chunk_size = CHUNK_SIZE as f32;
        let mut visible_chunks: Vec<(&ChunkPos, &LoadedChunk, f32)> = self
            .chunks
            .iter()
            .filter_map(|(pos, loaded)| {
                // Compute chunk AABB in world space
                let min = Vec3::new(
                    pos.x as f32 * chunk_size,
                    pos.y as f32 * chunk_size,
                    pos.z as f32 * chunk_size,
                );
                let max = min + Vec3::splat(chunk_size);
                let aabb = Aabb::new(min, max);

                // Frustum cull
                if !frustum.test_aabb(&aabb) {
                    return None;
                }

                // Compute distance to camera (using chunk center)
                let center = aabb.center();
                let dist_sq = (center - camera_pos).length_squared();

                Some((pos, loaded, dist_sq))
            })
            .collect();

        // Sort by distance (front-to-back)
        visible_chunks.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        if visible_chunks.is_empty() {
            self.chunk_info_address = 0;
            self.dirty = false;
            return Ok(());
        }

        // Collect chunk infos in sorted order
        let infos: Vec<GpuChunkInfo> = visible_chunks
            .iter()
            .map(|(_, loaded, _)| loaded.info)
            .collect();

        // Create new buffer
        let buffer_size = (infos.len() * GpuChunkInfo::SIZE) as u64;
        let usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        let chunk_info_buffer = allocator.create_buffer(
            buffer_size,
            usage,
            MemoryLocation::CpuToGpu,
            "world_chunk_info_culled",
        )?;

        // Write data
        chunk_info_buffer.write::<u8>(bytemuck::cast_slice(&infos))?;

        // Get device address
        self.chunk_info_address = chunk_info_buffer.device_address(device);
        self.chunk_info_buffer = Some(chunk_info_buffer);
        self.dirty = false;

        Ok(())
    }

    /// Get the number of visible chunks after culling.
    ///
    /// Returns the count from the last `rebuild_chunk_info_buffer_culled` call.
    pub fn visible_chunk_count(&self) -> u32 {
        self.chunk_info_buffer
            .as_ref()
            .map_or(0, |b| (b.size / GpuChunkInfo::SIZE as u64) as u32)
    }

    /// Get push constants for rendering.
    ///
    /// The chunk count reflects the number of visible chunks after culling
    /// (if `rebuild_chunk_info_buffer_culled` was used).
    pub fn push_constants(
        &self,
        screen_width: u32,
        screen_height: u32,
        max_steps: u32,
    ) -> WorldRenderPushConstants {
        // Use actual buffer count (accounts for culling) rather than total loaded chunks
        let chunk_count = self.visible_chunk_count();
        WorldRenderPushConstants {
            screen_size: [screen_width, screen_height],
            max_steps,
            chunk_count,
            chunk_info_address: self.chunk_info_address,
        }
    }

    /// Check if the renderer has any chunks to draw.
    pub fn has_chunks(&self) -> bool {
        !self.chunks.is_empty()
    }

    /// Get all loaded chunk positions.
    pub fn loaded_positions(&self) -> Vec<ChunkPos> {
        self.chunks.keys().copied().collect()
    }

    /// Destroy all GPU resources.
    pub fn destroy(mut self, allocator: &mut GpuAllocator) -> Result<()> {
        // Free chunk info buffer
        if let Some(mut buffer) = self.chunk_info_buffer.take() {
            allocator.free_buffer(&mut buffer)?;
        }

        // Free all chunk DAG buffers
        for (_, mut loaded) in self.chunks.drain() {
            allocator.free_buffer(&mut loaded.gpu_dag.node_buffer)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_chunk_info_size() {
        // Verify size is 32 bytes for good alignment
        assert_eq!(GpuChunkInfo::SIZE, 32);
    }

    #[test]
    fn gpu_chunk_info_layout() {
        // Verify offsets for shader compatibility
        assert_eq!(std::mem::offset_of!(GpuChunkInfo, node_buffer_address), 0);
        assert_eq!(std::mem::offset_of!(GpuChunkInfo, root_index), 8);
        assert_eq!(std::mem::offset_of!(GpuChunkInfo, octree_depth), 12);
        assert_eq!(std::mem::offset_of!(GpuChunkInfo, world_offset_x), 16);
        assert_eq!(std::mem::offset_of!(GpuChunkInfo, world_offset_y), 20);
        assert_eq!(std::mem::offset_of!(GpuChunkInfo, world_offset_z), 24);
        assert_eq!(std::mem::offset_of!(GpuChunkInfo, _padding), 28);
    }

    #[test]
    fn push_constants_size() {
        // Push constants should be 24 bytes
        assert_eq!(WorldRenderPushConstants::SIZE, 24);
    }

    #[test]
    fn push_constants_layout() {
        assert_eq!(
            std::mem::offset_of!(WorldRenderPushConstants, screen_size),
            0
        );
        assert_eq!(std::mem::offset_of!(WorldRenderPushConstants, max_steps), 8);
        assert_eq!(
            std::mem::offset_of!(WorldRenderPushConstants, chunk_count),
            12
        );
        assert_eq!(
            std::mem::offset_of!(WorldRenderPushConstants, chunk_info_address),
            16
        );
    }
}
