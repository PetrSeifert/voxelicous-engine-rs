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
use voxelicous_gpu::deferred::DeferredDeletionQueue;
use voxelicous_gpu::error::Result;
use voxelicous_gpu::memory::{GpuAllocator, GpuBuffer};
use voxelicous_voxel::{SvoDag, VoxelStorage};

use crate::debug::DebugMode;
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
    /// Debug visualization mode.
    pub debug_mode: u32,
    /// Padding for 8-byte alignment.
    pub _padding: u32,
}

impl WorldRenderPushConstants {
    /// Size in bytes.
    pub const SIZE: u32 = std::mem::size_of::<Self>() as u32;
}

/// Manages GPU resources for multi-chunk world rendering.
pub struct WorldRenderer {
    /// Per-chunk GPU data indexed by position.
    chunks: HashMap<ChunkPos, LoadedChunk>,
    /// Per-frame GPU buffers containing chunk info arrays for shader access.
    chunk_info_buffers: Vec<Option<GpuBuffer>>,
    /// Per-frame device addresses of chunk info buffers.
    chunk_info_addresses: Vec<vk::DeviceAddress>,
    /// Per-frame visible chunk counts after culling.
    visible_chunk_counts: Vec<u32>,
    /// Queue for deferred buffer deletions.
    deferred_deletions: DeferredDeletionQueue,
    /// Number of frames in flight.
    #[allow(dead_code)]
    frames_in_flight: usize,
    /// Maximum chunks to keep on GPU.
    #[allow(dead_code)]
    max_gpu_chunks: usize,
    /// Whether the chunk info buffer needs rebuilding.
    dirty: bool,
}

impl WorldRenderer {
    /// Create a new world renderer.
    ///
    /// # Arguments
    /// * `max_gpu_chunks` - Maximum number of chunks to keep on GPU.
    /// * `frames_in_flight` - Number of frames that can be in flight simultaneously.
    pub fn new(max_gpu_chunks: usize, frames_in_flight: usize) -> Self {
        Self {
            chunks: HashMap::with_capacity(max_gpu_chunks),
            chunk_info_buffers: (0..frames_in_flight).map(|_| None).collect(),
            chunk_info_addresses: vec![0; frames_in_flight],
            visible_chunk_counts: vec![0; frames_in_flight],
            deferred_deletions: DeferredDeletionQueue::new(frames_in_flight),
            frames_in_flight,
            max_gpu_chunks,
            dirty: true,
        }
    }

    /// Upload a chunk's DAG to the GPU.
    ///
    /// If a chunk already exists at this position, the old buffer is queued for deferred deletion.
    /// Returns `Ok(true)` if the chunk was uploaded, `Ok(false)` if skipped (empty/air-only).
    ///
    /// # Arguments
    /// * `allocator` - GPU allocator for creating buffers.
    /// * `device` - Vulkan device handle.
    /// * `pos` - Chunk position.
    /// * `dag` - The SVO-DAG to upload.
    /// * `frame_number` - Current frame number for deferred deletion tracking.
    pub fn upload_chunk(
        &mut self,
        allocator: &mut GpuAllocator,
        device: &ash::Device,
        pos: ChunkPos,
        dag: &SvoDag,
        frame_number: u64,
    ) -> Result<bool> {
        // Skip empty (air-only) chunks - no point uploading them
        if dag.is_empty() {
            // Still remove old chunk if it existed (chunk might have been cleared)
            if let Some(old) = self.chunks.remove(&pos) {
                // Queue old buffer for deferred deletion
                self.deferred_deletions.queue(old.gpu_dag.node_buffer, frame_number);
                self.dirty = true;
            }
            return Ok(false);
        }

        // Remove old chunk if it exists - queue for deferred deletion
        if let Some(old) = self.chunks.remove(&pos) {
            self.deferred_deletions.queue(old.gpu_dag.node_buffer, frame_number);
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
    /// The buffer is queued for deferred deletion rather than freed immediately.
    ///
    /// # Arguments
    /// * `pos` - Chunk position to remove.
    /// * `frame_number` - Current frame number for deferred deletion tracking.
    pub fn remove_chunk(&mut self, pos: ChunkPos, frame_number: u64) {
        if let Some(loaded) = self.chunks.remove(&pos) {
            self.deferred_deletions.queue(loaded.gpu_dag.node_buffer, frame_number);
            self.dirty = true;
        }
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
        let info_memory: u64 = self
            .chunk_info_buffers
            .iter()
            .filter_map(|b| b.as_ref())
            .map(|b| b.size)
            .sum();
        dag_memory + info_memory
    }

    /// Rebuild the chunk info buffer for shader access (non-culled version).
    ///
    /// Creates a per-frame buffer containing all chunks without frustum culling.
    /// Call this before rendering if chunks have changed.
    ///
    /// # Arguments
    /// * `allocator` - GPU allocator for creating buffers.
    /// * `device` - Vulkan device handle.
    /// * `frame_index` - Current frame index in the ring buffer.
    /// * `frame_number` - Current frame number for deferred deletion tracking.
    pub fn rebuild_chunk_info_buffer(
        &mut self,
        allocator: &mut GpuAllocator,
        device: &ash::Device,
        frame_index: usize,
        frame_number: u64,
    ) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }

        // Queue old buffer for deferred deletion if it exists
        if let Some(old_buffer) = self.chunk_info_buffers[frame_index].take() {
            self.deferred_deletions.queue(old_buffer, frame_number);
        }

        if self.chunks.is_empty() {
            self.chunk_info_addresses[frame_index] = 0;
            self.visible_chunk_counts[frame_index] = 0;
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
        self.chunk_info_addresses[frame_index] = chunk_info_buffer.device_address(device);
        self.visible_chunk_counts[frame_index] = infos.len() as u32;
        self.chunk_info_buffers[frame_index] = Some(chunk_info_buffer);
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
    /// # Arguments
    /// * `allocator` - GPU allocator for creating buffers.
    /// * `device` - Vulkan device handle.
    /// * `frustum` - Camera frustum for culling.
    /// * `camera_pos` - Camera position for distance sorting.
    /// * `frame_index` - Current frame index in the ring buffer.
    /// * `frame_number` - Current frame number for deferred deletion tracking.
    pub fn rebuild_chunk_info_buffer_culled(
        &mut self,
        allocator: &mut GpuAllocator,
        device: &ash::Device,
        frustum: &Frustum,
        camera_pos: Vec3,
        frame_index: usize,
        frame_number: u64,
    ) -> Result<()> {
        // Queue old buffer for deferred deletion if it exists
        if let Some(old_buffer) = self.chunk_info_buffers[frame_index].take() {
            self.deferred_deletions.queue(old_buffer, frame_number);
        }

        if self.chunks.is_empty() {
            self.chunk_info_addresses[frame_index] = 0;
            self.visible_chunk_counts[frame_index] = 0;
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
            self.chunk_info_addresses[frame_index] = 0;
            self.visible_chunk_counts[frame_index] = 0;
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
        self.chunk_info_addresses[frame_index] = chunk_info_buffer.device_address(device);
        self.visible_chunk_counts[frame_index] = infos.len() as u32;
        self.chunk_info_buffers[frame_index] = Some(chunk_info_buffer);
        self.dirty = false;

        Ok(())
    }

    /// Get the number of visible chunks after culling for a specific frame.
    ///
    /// Returns the count from the last `rebuild_chunk_info_buffer_culled` call for this frame.
    ///
    /// # Arguments
    /// * `frame_index` - Frame index to query.
    pub fn visible_chunk_count(&self, frame_index: usize) -> u32 {
        self.visible_chunk_counts[frame_index]
    }

    /// Get push constants for rendering.
    ///
    /// The chunk count reflects the number of visible chunks after culling
    /// (if `rebuild_chunk_info_buffer_culled` was used).
    ///
    /// # Arguments
    /// * `screen_width` - Screen width in pixels.
    /// * `screen_height` - Screen height in pixels.
    /// * `max_steps` - Maximum ray traversal steps per chunk.
    /// * `frame_index` - Current frame index in the ring buffer.
    /// * `debug_mode` - Debug visualization mode.
    pub fn push_constants(
        &self,
        screen_width: u32,
        screen_height: u32,
        max_steps: u32,
        frame_index: usize,
        debug_mode: DebugMode,
    ) -> WorldRenderPushConstants {
        WorldRenderPushConstants {
            screen_size: [screen_width, screen_height],
            max_steps,
            chunk_count: self.visible_chunk_counts[frame_index],
            chunk_info_address: self.chunk_info_addresses[frame_index],
            debug_mode: debug_mode.as_u32(),
            _padding: 0,
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

    /// Process deferred deletions.
    ///
    /// Call this at the start of each frame to free resources from completed frames.
    ///
    /// # Arguments
    /// * `allocator` - GPU allocator for freeing buffers.
    /// * `frame_number` - Current frame number.
    pub fn process_deferred_deletions(
        &mut self,
        allocator: &mut GpuAllocator,
        frame_number: u64,
    ) -> Result<()> {
        self.deferred_deletions.process(allocator, frame_number)
    }

    /// Destroy all GPU resources.
    ///
    /// Call this after `device_wait_idle()` to ensure all resources are freed safely.
    pub fn destroy(mut self, allocator: &mut GpuAllocator) -> Result<()> {
        // Flush all pending deferred deletions
        self.deferred_deletions.flush(allocator)?;

        // Free all per-frame chunk info buffers
        for buffer_opt in &mut self.chunk_info_buffers {
            if let Some(mut buffer) = buffer_opt.take() {
                allocator.free_buffer(&mut buffer)?;
            }
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
        // Push constants should be 32 bytes (24 + debug_mode + padding)
        assert_eq!(WorldRenderPushConstants::SIZE, 32);
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
        assert_eq!(
            std::mem::offset_of!(WorldRenderPushConstants, debug_mode),
            24
        );
        assert_eq!(
            std::mem::offset_of!(WorldRenderPushConstants, _padding),
            28
        );
    }
}
