//! Clipmap GPU resources and upload helpers.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;
use voxelicous_gpu::deferred::DeferredDeletionQueue;
use voxelicous_gpu::error::Result;
use voxelicous_gpu::memory::{GpuAllocator, GpuBuffer};
use voxelicous_voxel::{
    BrickHeader, BrickId, ClipmapVoxelStore, CLIPMAP_LOD_COUNT, CLIPMAP_PAGE_GRID, PAGE_BRICKS,
    PALETTE16_STRIDE, PALETTE32_STRIDE, RAW16_STRIDE,
};
use voxelicous_world::{ClipmapDirtyState, ClipmapStreamingController};

use crate::debug::DebugMode;

/// GPU-side clipmap info shared with the shader (buffer reference).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuClipmapInfo {
    pub page_brick_indices_addr: [u64; CLIPMAP_LOD_COUNT],
    pub page_occ_addr: [u64; CLIPMAP_LOD_COUNT],
    pub page_coord_addr: [u64; CLIPMAP_LOD_COUNT],
    pub brick_header_addr: u64,
    pub palette16_addr: u64,
    pub palette32_addr: u64,
    pub raw16_addr: u64,
    pub _pad0: [u64; 2],
    pub origin: [[i32; 4]; CLIPMAP_LOD_COUNT],
    pub voxel_size: [[u32; 4]; CLIPMAP_LOD_COUNT],
    pub lod_aabb_min: [[f32; 4]; CLIPMAP_LOD_COUNT],
    pub lod_aabb_max: [[f32; 4]; CLIPMAP_LOD_COUNT],
}

impl GpuClipmapInfo {
    pub const SIZE: usize = std::mem::size_of::<Self>();
}

/// Push constants for clipmap ray marching.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ClipmapRenderPushConstants {
    pub screen_size: [u32; 2],
    pub max_steps: u32,
    pub _pad0: u32,
    pub clipmap_info_address: u64,
    pub debug_mode: u32,
    pub _pad1: u32,
}

impl ClipmapRenderPushConstants {
    pub const SIZE: u32 = std::mem::size_of::<Self>() as u32;
}

/// GPU resources for clipmap rendering.
pub struct ClipmapRenderer {
    page_brick_buffers: Vec<Option<GpuBuffer>>,
    page_occ_buffers: Vec<Option<GpuBuffer>>,
    page_coord_buffers: Vec<Option<GpuBuffer>>,
    brick_header_buffer: Option<GpuBuffer>,
    palette16_buffer: Option<GpuBuffer>,
    palette32_buffer: Option<GpuBuffer>,
    raw16_buffer: Option<GpuBuffer>,
    clipmap_info_buffers: Vec<Option<GpuBuffer>>,
    clipmap_info_addresses: Vec<vk::DeviceAddress>,
    deferred_deletions: DeferredDeletionQueue,
    frames_in_flight: usize,
}

impl ClipmapRenderer {
    /// Create a new clipmap renderer.
    pub fn new(frames_in_flight: usize) -> Self {
        let mut page_brick_buffers = Vec::with_capacity(CLIPMAP_LOD_COUNT);
        page_brick_buffers.resize_with(CLIPMAP_LOD_COUNT, || None);
        let mut page_occ_buffers = Vec::with_capacity(CLIPMAP_LOD_COUNT);
        page_occ_buffers.resize_with(CLIPMAP_LOD_COUNT, || None);
        let mut page_coord_buffers = Vec::with_capacity(CLIPMAP_LOD_COUNT);
        page_coord_buffers.resize_with(CLIPMAP_LOD_COUNT, || None);
        let mut clipmap_info_buffers = Vec::with_capacity(frames_in_flight);
        clipmap_info_buffers.resize_with(frames_in_flight, || None);

        Self {
            page_brick_buffers,
            page_occ_buffers,
            page_coord_buffers,
            brick_header_buffer: None,
            palette16_buffer: None,
            palette32_buffer: None,
            raw16_buffer: None,
            clipmap_info_buffers,
            clipmap_info_addresses: vec![0; frames_in_flight],
            deferred_deletions: DeferredDeletionQueue::new(frames_in_flight),
            frames_in_flight,
        }
    }

    /// Ensure all GPU buffers exist and are large enough.
    #[cfg_attr(
        feature = "profiling-tracy",
        tracing::instrument(level = "trace", skip_all)
    )]
    pub fn sync_from_controller(
        &mut self,
        allocator: &mut GpuAllocator,
        device: &ash::Device,
        controller: &ClipmapStreamingController,
        dirty: ClipmapDirtyState,
        frame_index: usize,
        frame_number: u64,
    ) -> Result<()> {
        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.ensure_page_buffers").entered();
            self.ensure_page_buffers(allocator, controller.active_lod_count(), frame_number)?;
        }
        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.ensure_info_buffers").entered();
            self.ensure_info_buffers(allocator, device)?;
        }

        let store = controller.store();

        let header_realloc = {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.ensure_brick_header_buffer").entered();
            self.ensure_brick_header_buffer(allocator, frame_number, store.headers())?
        };
        let pal16_realloc = {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.ensure_palette16_buffer").entered();
            Self::ensure_pool_buffer(
                allocator,
                &mut self.deferred_deletions,
                frame_number,
                &mut self.palette16_buffer,
                store.palette16_pool().len() as u64,
                PALETTE16_STRIDE as u64,
                "clipmap_palette16",
            )?
        };
        let pal32_realloc = {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.ensure_palette32_buffer").entered();
            Self::ensure_pool_buffer(
                allocator,
                &mut self.deferred_deletions,
                frame_number,
                &mut self.palette32_buffer,
                store.palette32_pool().len() as u64,
                PALETTE32_STRIDE as u64,
                "clipmap_palette32",
            )?
        };
        let raw_realloc = {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.ensure_raw16_buffer").entered();
            Self::ensure_pool_buffer(
                allocator,
                &mut self.deferred_deletions,
                frame_number,
                &mut self.raw16_buffer,
                store.raw16_pool().len() as u64,
                RAW16_STRIDE as u64,
                "clipmap_raw16",
            )?
        };

        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.upload_page_tables").entered();
            self.upload_page_tables(controller, dirty.dirty_pages)?;
        }
        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.upload_brick_headers").entered();
            self.upload_brick_headers(store, dirty.dirty_headers, header_realloc)?;
        }
        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.upload_palette16_entries").entered();
            self.upload_pool_entries(
                store.palette16_pool(),
                PALETTE16_STRIDE,
                self.palette16_buffer.as_ref().unwrap(),
                dirty.dirty_palette16_entries,
                pal16_realloc,
            )?;
        }
        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.upload_palette32_entries").entered();
            self.upload_pool_entries(
                store.palette32_pool(),
                PALETTE32_STRIDE,
                self.palette32_buffer.as_ref().unwrap(),
                dirty.dirty_palette32_entries,
                pal32_realloc,
            )?;
        }
        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.upload_raw16_entries").entered();
            self.upload_pool_entries(
                store.raw16_pool(),
                RAW16_STRIDE,
                self.raw16_buffer.as_ref().unwrap(),
                dirty.dirty_raw16_entries,
                raw_realloc,
            )?;
        }

        let info = {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.build_gpu_info").entered();
            self.build_gpu_info(device, controller)
        };
        if let Some(info_buffer) = &self.clipmap_info_buffers[frame_index] {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.write_gpu_info").entered();
            info_buffer.write(std::slice::from_ref(&info))?;
            self.clipmap_info_addresses[frame_index] = info_buffer.device_address(device);
        }

        Ok(())
    }

    /// Get push constants for rendering.
    pub fn push_constants(
        &self,
        screen_width: u32,
        screen_height: u32,
        max_steps: u32,
        frame_index: usize,
        debug_mode: DebugMode,
    ) -> ClipmapRenderPushConstants {
        ClipmapRenderPushConstants {
            screen_size: [screen_width, screen_height],
            max_steps,
            _pad0: 0,
            clipmap_info_address: self.clipmap_info_addresses[frame_index],
            debug_mode: debug_mode.as_u32(),
            _pad1: 0,
        }
    }

    /// Process deferred deletions.
    pub fn process_deferred_deletions(
        &mut self,
        allocator: &mut GpuAllocator,
        frame_number: u64,
    ) -> Result<()> {
        self.deferred_deletions.process(allocator, frame_number)
    }

    /// Destroy all GPU resources.
    pub fn destroy(mut self, allocator: &mut GpuAllocator) -> Result<()> {
        self.deferred_deletions.flush(allocator)?;

        for buffer in &mut self.page_brick_buffers {
            if let Some(mut buf) = buffer.take() {
                allocator.free_buffer(&mut buf)?;
            }
        }
        for buffer in &mut self.page_occ_buffers {
            if let Some(mut buf) = buffer.take() {
                allocator.free_buffer(&mut buf)?;
            }
        }
        for buffer in &mut self.page_coord_buffers {
            if let Some(mut buf) = buffer.take() {
                allocator.free_buffer(&mut buf)?;
            }
        }
        for buffer in &mut self.clipmap_info_buffers {
            if let Some(mut buf) = buffer.take() {
                allocator.free_buffer(&mut buf)?;
            }
        }
        if let Some(mut buf) = self.brick_header_buffer.take() {
            allocator.free_buffer(&mut buf)?;
        }
        if let Some(mut buf) = self.palette16_buffer.take() {
            allocator.free_buffer(&mut buf)?;
        }
        if let Some(mut buf) = self.palette32_buffer.take() {
            allocator.free_buffer(&mut buf)?;
        }
        if let Some(mut buf) = self.raw16_buffer.take() {
            allocator.free_buffer(&mut buf)?;
        }

        Ok(())
    }

    fn ensure_page_buffers(
        &mut self,
        allocator: &mut GpuAllocator,
        active_lod_count: usize,
        frame_number: u64,
    ) -> Result<()> {
        let page_count = CLIPMAP_PAGE_GRID * CLIPMAP_PAGE_GRID * CLIPMAP_PAGE_GRID;
        let brick_bytes = (page_count * PAGE_BRICKS * std::mem::size_of::<u32>()) as u64;
        let occ_bytes = (page_count * 2 * std::mem::size_of::<u32>()) as u64;
        let coord_bytes = (page_count * std::mem::size_of::<[i32; 4]>()) as u64;
        let usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        for lod in 0..active_lod_count {
            if self.page_brick_buffers[lod].is_none() {
                let buffer = allocator.create_buffer(
                    brick_bytes,
                    usage,
                    MemoryLocation::CpuToGpu,
                    &format!("clipmap_page_bricks_lod{lod}"),
                )?;
                self.page_brick_buffers[lod] = Some(buffer);
            }
            if self.page_occ_buffers[lod].is_none() {
                let buffer = allocator.create_buffer(
                    occ_bytes,
                    usage,
                    MemoryLocation::CpuToGpu,
                    &format!("clipmap_page_occ_lod{lod}"),
                )?;
                self.page_occ_buffers[lod] = Some(buffer);
            }
            if self.page_coord_buffers[lod].is_none() {
                let buffer = allocator.create_buffer(
                    coord_bytes,
                    usage,
                    MemoryLocation::CpuToGpu,
                    &format!("clipmap_page_coord_lod{lod}"),
                )?;
                self.page_coord_buffers[lod] = Some(buffer);
            }
        }

        for lod in active_lod_count..CLIPMAP_LOD_COUNT {
            if let Some(buffer) = self.page_brick_buffers[lod].take() {
                self.deferred_deletions.queue(buffer, frame_number);
            }
            if let Some(buffer) = self.page_occ_buffers[lod].take() {
                self.deferred_deletions.queue(buffer, frame_number);
            }
            if let Some(buffer) = self.page_coord_buffers[lod].take() {
                self.deferred_deletions.queue(buffer, frame_number);
            }
        }

        Ok(())
    }

    fn ensure_info_buffers(
        &mut self,
        allocator: &mut GpuAllocator,
        device: &ash::Device,
    ) -> Result<()> {
        if self.clipmap_info_buffers.iter().all(|b| b.is_some()) {
            return Ok(());
        }

        let usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        for i in 0..self.frames_in_flight {
            if self.clipmap_info_buffers[i].is_none() {
                let buffer = allocator.create_buffer(
                    GpuClipmapInfo::SIZE as u64,
                    usage,
                    MemoryLocation::CpuToGpu,
                    &format!("clipmap_info_{i}"),
                )?;
                self.clipmap_info_addresses[i] = buffer.device_address(device);
                self.clipmap_info_buffers[i] = Some(buffer);
            }
        }

        Ok(())
    }

    fn ensure_brick_header_buffer(
        &mut self,
        allocator: &mut GpuAllocator,
        frame_number: u64,
        headers: &[BrickHeader],
    ) -> Result<bool> {
        let required = (headers.len() * std::mem::size_of::<BrickHeader>()) as u64;
        let usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let buffer = &mut self.brick_header_buffer;

        if buffer.as_ref().map_or(true, |b| b.size < required) {
            let current_size = buffer.as_ref().map_or(0, |b| b.size);
            if let Some(old) = buffer.take() {
                self.deferred_deletions.queue(old, frame_number);
            }
            let min_size = std::mem::size_of::<BrickHeader>() as u64;
            let required_size = required.max(min_size);
            let size = if current_size == 0 {
                required_size
            } else {
                current_size.saturating_mul(2).max(required_size)
            };
            let new_buffer = allocator.create_buffer(
                size,
                usage,
                MemoryLocation::CpuToGpu,
                "clipmap_brick_headers",
            )?;
            *buffer = Some(new_buffer);
            return Ok(true);
        }

        Ok(false)
    }

    fn ensure_pool_buffer(
        allocator: &mut GpuAllocator,
        deferred: &mut DeferredDeletionQueue,
        frame_number: u64,
        buffer: &mut Option<GpuBuffer>,
        pool_size: u64,
        stride: u64,
        name: &str,
    ) -> Result<bool> {
        let required = pool_size.max(stride);
        let usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        if buffer.as_ref().map_or(true, |b| b.size < required) {
            let current_size = buffer.as_ref().map_or(0, |b| b.size);
            if let Some(old) = buffer.take() {
                deferred.queue(old, frame_number);
            }
            let grow_size = if current_size == 0 {
                required
            } else {
                current_size.saturating_mul(2).max(required)
            };
            let new_buffer =
                allocator.create_buffer(grow_size, usage, MemoryLocation::CpuToGpu, name)?;
            *buffer = Some(new_buffer);
            return Ok(true);
        }

        Ok(false)
    }

    #[cfg_attr(
        feature = "profiling-tracy",
        tracing::instrument(level = "trace", skip_all)
    )]
    fn upload_page_tables(
        &self,
        controller: &ClipmapStreamingController,
        dirty_pages: Vec<Vec<usize>>,
    ) -> Result<()> {
        let page_count = CLIPMAP_PAGE_GRID * CLIPMAP_PAGE_GRID * CLIPMAP_PAGE_GRID;

        for lod in 0..CLIPMAP_LOD_COUNT {
            let Some(brick_buffer) = &self.page_brick_buffers[lod] else {
                continue;
            };
            let Some(occ_buffer) = &self.page_occ_buffers[lod] else {
                continue;
            };
            let Some(coord_buffer) = &self.page_coord_buffers[lod] else {
                continue;
            };

            let page_bricks = controller.page_brick_indices(lod);
            let page_occ = controller.page_occ(lod);
            let page_coords = controller.page_coords(lod);

            let Some(lod_dirty_pages) = dirty_pages.get(lod) else {
                continue;
            };
            if lod_dirty_pages.is_empty() {
                // No dirty pages; skip.
                continue;
            }

            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!(
                "upload_page_tables_lod",
                lod = lod as u32,
                dirty_pages = lod_dirty_pages.len() as u32
            )
            .entered();

            for &page_index in lod_dirty_pages {
                if page_index >= page_count {
                    continue;
                }
                let base = page_index * PAGE_BRICKS;
                let offset = (base * std::mem::size_of::<u32>()) as u64;
                brick_buffer.write_range(offset, &page_bricks[base..base + PAGE_BRICKS])?;

                let occ_offset = (page_index * 2 * std::mem::size_of::<u32>()) as u64;
                occ_buffer.write_range(occ_offset, &page_occ[page_index])?;

                let coord_offset = (page_index * std::mem::size_of::<[i32; 4]>()) as u64;
                coord_buffer.write_range(coord_offset, &page_coords[page_index])?;
            }
        }

        Ok(())
    }

    #[cfg_attr(
        feature = "profiling-tracy",
        tracing::instrument(level = "trace", skip_all)
    )]
    fn upload_brick_headers(
        &self,
        store: &ClipmapVoxelStore,
        dirty_headers: Vec<BrickId>,
        full_upload: bool,
    ) -> Result<()> {
        let Some(header_buffer) = &self.brick_header_buffer else {
            return Ok(());
        };
        let headers = store.headers();

        if full_upload {
            #[cfg(feature = "profiling-tracy")]
            let _span =
                tracing::trace_span!("upload_brick_headers_full", headers = headers.len() as u32)
                    .entered();
            header_buffer.write(headers)?;
            return Ok(());
        }

        let header_size = std::mem::size_of::<BrickHeader>();
        #[cfg(feature = "profiling-tracy")]
        let _span = tracing::trace_span!(
            "upload_brick_headers_incremental",
            dirty_headers = dirty_headers.len() as u32
        )
        .entered();
        for id in dirty_headers {
            let idx = id.0 as usize;
            if idx >= headers.len() {
                continue;
            }
            let offset = (idx * header_size) as u64;
            header_buffer.write_range(offset, &headers[idx..idx + 1])?;
        }

        Ok(())
    }

    #[cfg_attr(
        feature = "profiling-tracy",
        tracing::instrument(level = "trace", skip_all)
    )]
    fn upload_pool_entries(
        &self,
        pool: &[u8],
        stride: usize,
        buffer: &GpuBuffer,
        entries: Vec<u32>,
        full_upload: bool,
    ) -> Result<()> {
        if pool.is_empty() {
            return Ok(());
        }

        if full_upload {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("upload_pool_entries_full", bytes = pool.len() as u32)
                .entered();
            buffer.write_bytes(0, pool)?;
            return Ok(());
        }

        #[cfg(feature = "profiling-tracy")]
        let _span = tracing::trace_span!(
            "upload_pool_entries_incremental",
            entries = entries.len() as u32,
            stride = stride as u32
        )
        .entered();
        for entry in entries {
            let offset = entry as usize * stride;
            if offset + stride <= pool.len() {
                buffer.write_bytes(offset as u64, &pool[offset..offset + stride])?;
            }
        }

        Ok(())
    }

    #[cfg_attr(
        feature = "profiling-tracy",
        tracing::instrument(level = "trace", skip_all)
    )]
    fn build_gpu_info(
        &self,
        device: &ash::Device,
        controller: &ClipmapStreamingController,
    ) -> GpuClipmapInfo {
        let mut info = GpuClipmapInfo::zeroed();

        for lod in 0..CLIPMAP_LOD_COUNT {
            if let Some(buffer) = &self.page_brick_buffers[lod] {
                info.page_brick_indices_addr[lod] = buffer.device_address(device);
            }
            if let Some(buffer) = &self.page_occ_buffers[lod] {
                info.page_occ_addr[lod] = buffer.device_address(device);
            }
            if let Some(buffer) = &self.page_coord_buffers[lod] {
                info.page_coord_addr[lod] = buffer.device_address(device);
            }

            let origin = controller.lod_origin(lod);
            let renderable = controller.lod_renderable(lod);
            let voxel_size = if renderable {
                controller.lod_voxel_size(lod) as u32
            } else {
                0
            };
            let coverage = if renderable {
                controller.lod_coverage(lod) as f32
            } else {
                0.0
            };

            info.origin[lod] = [origin.x as i32, origin.y as i32, origin.z as i32, 0];
            info.voxel_size[lod] = [voxel_size, 0, 0, 0];
            info.lod_aabb_min[lod] = [origin.x as f32, origin.y as f32, origin.z as f32, 0.0];
            info.lod_aabb_max[lod] = [
                origin.x as f32 + coverage,
                origin.y as f32 + coverage,
                origin.z as f32 + coverage,
                0.0,
            ];
        }

        if let Some(buffer) = &self.brick_header_buffer {
            info.brick_header_addr = buffer.device_address(device);
        }
        if let Some(buffer) = &self.palette16_buffer {
            info.palette16_addr = buffer.device_address(device);
        }
        if let Some(buffer) = &self.palette32_buffer {
            info.palette32_addr = buffer.device_address(device);
        }
        if let Some(buffer) = &self.raw16_buffer {
            info.raw16_addr = buffer.device_address(device);
        }

        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_clipmap_info_size() {
        assert_eq!(GpuClipmapInfo::SIZE, 576);
    }

    #[test]
    fn push_constants_size() {
        assert_eq!(ClipmapRenderPushConstants::SIZE, 32);
    }
}
