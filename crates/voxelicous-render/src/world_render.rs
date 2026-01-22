//! Multi-chunk world rendering for procedural terrain.
//!
//! Manages GPU resources for rendering multiple SVO-DAG chunks simultaneously.
//! Uses deferred resource deletion to avoid GPU synchronization stalls.

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

/// Default number of frames to wait before deleting resources.
const DEFAULT_FRAMES_IN_FLIGHT: u64 = 3;

/// A buffer queued for deferred deletion.
///
/// Instead of calling `device_wait_idle()` before freeing a buffer,
/// we track when it was last used and delete it after enough frames pass.
struct DeferredDeletion {
    /// The buffer to be deleted.
    buffer: GpuBuffer,
    /// Frame number when the buffer was last potentially in use.
    frame_submitted: u64,
}

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
///
/// Uses deferred resource deletion to avoid GPU synchronization stalls.
/// Instead of calling `device_wait_idle()` before freeing buffers, buffers
/// are queued for deletion and freed after `frames_in_flight` frames pass.
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
    /// Buffers pending deferred deletion.
    pending_deletions: Vec<DeferredDeletion>,
    /// Current frame number (incremented each frame).
    current_frame: u64,
    /// Number of frames to wait before deleting resources.
    frames_in_flight: u64,
}

impl WorldRenderer {
    /// Create a new world renderer.
    pub fn new(max_gpu_chunks: usize) -> Self {
        Self::with_frames_in_flight(max_gpu_chunks, DEFAULT_FRAMES_IN_FLIGHT)
    }

    /// Create a new world renderer with custom frames-in-flight setting.
    ///
    /// `frames_in_flight` controls how many frames to wait before freeing
    /// GPU resources. Higher values increase memory usage but reduce the
    /// chance of freeing resources still in use by the GPU.
    pub fn with_frames_in_flight(max_gpu_chunks: usize, frames_in_flight: u64) -> Self {
        Self {
            chunks: HashMap::with_capacity(max_gpu_chunks),
            chunk_info_buffer: None,
            chunk_info_address: 0,
            max_gpu_chunks,
            dirty: true,
            pending_deletions: Vec::new(),
            current_frame: 0,
            frames_in_flight,
        }
    }

    /// Queue a buffer for deferred deletion.
    ///
    /// The buffer will be freed after `frames_in_flight` frames have passed
    /// to ensure it's no longer in use by the GPU.
    fn queue_deletion(&mut self, buffer: GpuBuffer) {
        self.pending_deletions.push(DeferredDeletion {
            buffer,
            frame_submitted: self.current_frame,
        });
    }

    /// Process deferred deletions, freeing buffers that are safe to delete.
    ///
    /// Call this at the start of each frame before uploading new resources.
    pub fn process_deferred_deletions(&mut self, allocator: &mut GpuAllocator) -> Result<()> {
        // Partition: keep buffers still in flight, collect ones ready to free
        let threshold = self.current_frame.saturating_sub(self.frames_in_flight);

        let mut i = 0;
        while i < self.pending_deletions.len() {
            if self.pending_deletions[i].frame_submitted <= threshold {
                let mut deletion = self.pending_deletions.swap_remove(i);
                allocator.free_buffer(&mut deletion.buffer)?;
                // Don't increment i since swap_remove moved last element here
            } else {
                i += 1;
            }
        }

        Ok(())
    }

    /// Advance to the next frame.
    ///
    /// Call this once per frame, typically at the end of rendering.
    pub fn advance_frame(&mut self) {
        self.current_frame = self.current_frame.wrapping_add(1);
    }

    /// Get the current frame number.
    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Get the number of pending deletions.
    pub fn pending_deletion_count(&self) -> usize {
        self.pending_deletions.len()
    }

    /// Upload a chunk's DAG to the GPU.
    ///
    /// If a chunk already exists at this position, the old buffer is queued for
    /// deferred deletion to avoid GPU synchronization stalls.
    ///
    /// Returns `Ok(true)` if the chunk was uploaded, `Ok(false)` if skipped (empty/air-only).
    ///
    /// **Important**: Call `process_deferred_deletions()` at the start of each frame
    /// to free buffers that are no longer in use.
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
                // Queue old buffer for deferred deletion instead of immediate free
                self.queue_deletion(old.gpu_dag.node_buffer);
                self.dirty = true;
            }
            return Ok(false);
        }

        // Remove old chunk if it exists - queue for deferred deletion
        if let Some(old) = self.chunks.remove(&pos) {
            self.queue_deletion(old.gpu_dag.node_buffer);
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
    /// The buffer is queued for deferred deletion to avoid GPU stalls.
    ///
    /// **Note**: The `allocator` and `device` parameters are retained for API
    /// compatibility but are no longer used. Call `process_deferred_deletions()`
    /// at frame start to free queued buffers.
    #[allow(unused_variables)]
    pub fn remove_chunk(
        &mut self,
        allocator: &mut GpuAllocator,
        device: &ash::Device,
        pos: ChunkPos,
    ) -> Result<()> {
        if let Some(loaded) = self.chunks.remove(&pos) {
            self.queue_deletion(loaded.gpu_dag.node_buffer);
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
    /// **Important**: Call `process_deferred_deletions()` at the start of each frame
    /// to free old chunk info buffers.
    pub fn rebuild_chunk_info_buffer(
        &mut self,
        allocator: &mut GpuAllocator,
        device: &ash::Device,
    ) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }

        // Queue old buffer for deferred deletion (if any)
        if let Some(old_buffer) = self.chunk_info_buffer.take() {
            self.queue_deletion(old_buffer);
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
    /// **Note**: Fast camera rotation may cause visible pop-in as chunks enter
    /// the frustum. Use `rebuild_chunk_info_buffer_sorted()` to include all
    /// loaded chunks without culling (relies on shader early-exit instead).
    ///
    /// **Important**: Call `process_deferred_deletions()` at the start of each frame
    /// to free old chunk info buffers.
    pub fn rebuild_chunk_info_buffer_culled(
        &mut self,
        allocator: &mut GpuAllocator,
        device: &ash::Device,
        frustum: &Frustum,
        camera_pos: Vec3,
    ) -> Result<()> {
        // Always rebuild when using culling since visibility changes with camera
        // Queue old buffer for deferred deletion (if any)
        if let Some(old_buffer) = self.chunk_info_buffer.take() {
            self.queue_deletion(old_buffer);
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

    /// Rebuild the chunk info buffer with distance sorting but NO frustum culling.
    ///
    /// This version includes ALL loaded chunks sorted front-to-back by distance.
    /// Unlike `rebuild_chunk_info_buffer_culled()`, this prevents pop-in during
    /// fast camera rotation by including all chunks regardless of visibility.
    ///
    /// The shader's early AABB test will skip chunks that don't intersect the ray,
    /// so performance impact is minimal when combined with front-to-back ordering.
    ///
    /// **Important**: Call `process_deferred_deletions()` at the start of each frame
    /// to free old chunk info buffers.
    pub fn rebuild_chunk_info_buffer_sorted(
        &mut self,
        allocator: &mut GpuAllocator,
        device: &ash::Device,
        camera_pos: Vec3,
    ) -> Result<()> {
        // Queue old buffer for deferred deletion (if any)
        if let Some(old_buffer) = self.chunk_info_buffer.take() {
            self.queue_deletion(old_buffer);
        }

        if self.chunks.is_empty() {
            self.chunk_info_address = 0;
            self.dirty = false;
            return Ok(());
        }

        // Collect ALL chunks with distances (no frustum culling)
        let chunk_size = CHUNK_SIZE as f32;
        let mut all_chunks: Vec<(&ChunkPos, &LoadedChunk, f32)> = self
            .chunks
            .iter()
            .map(|(pos, loaded)| {
                // Compute chunk center in world space
                let center = Vec3::new(
                    (pos.x as f32 + 0.5) * chunk_size,
                    (pos.y as f32 + 0.5) * chunk_size,
                    (pos.z as f32 + 0.5) * chunk_size,
                );
                let dist_sq = (center - camera_pos).length_squared();
                (pos, loaded, dist_sq)
            })
            .collect();

        // Sort by distance (front-to-back) for early shader termination
        all_chunks.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Collect chunk infos in sorted order
        let infos: Vec<GpuChunkInfo> = all_chunks
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
            "world_chunk_info_sorted",
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

    /// Flush all pending deletions immediately.
    ///
    /// This forces deletion of all queued buffers regardless of frame count.
    /// Useful before shutdown or when you know the GPU is idle.
    ///
    /// **Warning**: Only call this when you're certain the GPU has finished
    /// all work using these buffers (e.g., after `device_wait_idle()`).
    pub fn flush_pending_deletions(&mut self, allocator: &mut GpuAllocator) -> Result<()> {
        for mut deletion in self.pending_deletions.drain(..) {
            allocator.free_buffer(&mut deletion.buffer)?;
        }
        Ok(())
    }

    /// Destroy all GPU resources.
    ///
    /// Frees all buffers including those pending deletion.
    pub fn destroy(mut self, allocator: &mut GpuAllocator) -> Result<()> {
        // Free chunk info buffer
        if let Some(mut buffer) = self.chunk_info_buffer.take() {
            allocator.free_buffer(&mut buffer)?;
        }

        // Free all chunk DAG buffers
        for (_, mut loaded) in self.chunks.drain() {
            allocator.free_buffer(&mut loaded.gpu_dag.node_buffer)?;
        }

        // Free all pending deletions
        for mut deletion in self.pending_deletions.drain(..) {
            allocator.free_buffer(&mut deletion.buffer)?;
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

    #[test]
    fn world_renderer_default_frames_in_flight() {
        let renderer = WorldRenderer::new(100);
        // Default should be 3 frames in flight
        assert_eq!(renderer.frames_in_flight, DEFAULT_FRAMES_IN_FLIGHT);
        assert_eq!(renderer.current_frame, 0);
        assert_eq!(renderer.pending_deletion_count(), 0);
    }

    #[test]
    fn world_renderer_custom_frames_in_flight() {
        let renderer = WorldRenderer::with_frames_in_flight(100, 5);
        assert_eq!(renderer.frames_in_flight, 5);
    }

    #[test]
    fn world_renderer_advance_frame() {
        let mut renderer = WorldRenderer::new(100);
        assert_eq!(renderer.current_frame(), 0);
        renderer.advance_frame();
        assert_eq!(renderer.current_frame(), 1);
        renderer.advance_frame();
        assert_eq!(renderer.current_frame(), 2);
    }
}
