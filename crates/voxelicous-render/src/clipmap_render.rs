//! Clipmap GPU resources and upload helpers.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;
use voxelicous_gpu::error::Result;
use voxelicous_gpu::memory::{GpuAllocator, GpuBuffer};
use voxelicous_gpu::DeferredDeletionQueue;
use voxelicous_voxel::{
    BrickHeader, BrickId, ClipmapVoxelStore, CLIPMAP_LOD_COUNT, CLIPMAP_PAGE_GRID, PAGE_BRICKS,
    PAGE_VOXELS_PER_AXIS, PALETTE16_STRIDE, PALETTE32_STRIDE, RAW16_STRIDE,
};
use voxelicous_world::{ClipmapDirtyState, ClipmapStreamingController};

use crate::debug::DebugMode;

const INVALID_PAGE_COORD: [i32; 4] = [i32::MIN, i32::MIN, i32::MIN, 0];
const INIT_CHUNK_U32: usize = 16 * 1024;
const INIT_CHUNK_COORD: usize = 4 * 1024;

/// Growth factor for pool buffers
const POOL_GROWTH_FACTOR: u64 = 2;
const PALETTE16_CHUNK_ENTRIES: usize = 256 * 1024;
const PALETTE16_CHUNK_BYTES: u64 = (PALETTE16_CHUNK_ENTRIES * PALETTE16_STRIDE) as u64;

/// GPU-side clipmap info shared with the shader (buffer reference).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuClipmapInfo {
    pub page_brick_indices_addr: [u64; CLIPMAP_LOD_COUNT],
    pub page_occ_addr: [u64; CLIPMAP_LOD_COUNT],
    pub page_coord_addr: [u64; CLIPMAP_LOD_COUNT],
    pub brick_header_addr: u64,
    /// Address of a `uint64_t[]` table containing palette16 chunk buffer addresses.
    pub palette16_addr: u64,
    pub palette32_addr: u64,
    pub raw16_addr: u64,
    pub _pad0: [u64; 2],
    /// Per-LOD world page coordinate containing the camera-relative render origin.
    pub origin: [[i32; 4]; CLIPMAP_LOD_COUNT],
    /// x = LOD voxel size, yzw = render-origin offset inside that LOD's page.
    pub voxel_size: [[u32; 4]; CLIPMAP_LOD_COUNT],
    /// Camera-relative LOD bounds used by the shader.
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

fn entries_for_pool_size(pool_size: u64, stride: u64) -> usize {
    if pool_size == 0 {
        0
    } else {
        pool_size.div_ceil(stride) as usize
    }
}

fn div_floor(value: i64, divisor: i64) -> i64 {
    let mut q = value / divisor;
    let r = value % divisor;
    if r != 0 && ((r > 0) != (divisor > 0)) {
        q -= 1;
    }
    q
}

struct FrameBuffers {
    page_brick_buffers: Vec<Option<GpuBuffer>>,
    page_occ_buffers: Vec<Option<GpuBuffer>>,
    page_coord_buffers: Vec<Option<GpuBuffer>>,
    brick_header_buffer: Option<GpuBuffer>,
    palette16_pool: Palette16GpuPool,
    palette32_buffer: Option<GpuBuffer>,
    raw16_buffer: Option<GpuBuffer>,
    clipmap_info_buffer: Option<GpuBuffer>,
}

impl FrameBuffers {
    fn new() -> Self {
        let mut page_brick_buffers = Vec::with_capacity(CLIPMAP_LOD_COUNT);
        page_brick_buffers.resize_with(CLIPMAP_LOD_COUNT, || None);
        let mut page_occ_buffers = Vec::with_capacity(CLIPMAP_LOD_COUNT);
        page_occ_buffers.resize_with(CLIPMAP_LOD_COUNT, || None);
        let mut page_coord_buffers = Vec::with_capacity(CLIPMAP_LOD_COUNT);
        page_coord_buffers.resize_with(CLIPMAP_LOD_COUNT, || None);

        Self {
            page_brick_buffers,
            page_occ_buffers,
            page_coord_buffers,
            brick_header_buffer: None,
            palette16_pool: Palette16GpuPool::new(),
            palette32_buffer: None,
            raw16_buffer: None,
            clipmap_info_buffer: None,
        }
    }
}

struct Palette16GpuPool {
    chunks: Vec<GpuBuffer>,
    address_table: Option<GpuBuffer>,
    table_capacity: usize,
}

impl Palette16GpuPool {
    fn new() -> Self {
        Self {
            chunks: Vec::new(),
            address_table: None,
            table_capacity: 0,
        }
    }
}

struct PendingDirtyState {
    dirty_pages: Vec<Vec<usize>>,
    dirty_headers: Vec<BrickId>,
    dirty_palette16_entries: Vec<u32>,
    dirty_palette32_entries: Vec<u32>,
    dirty_raw16_entries: Vec<u32>,
}

impl PendingDirtyState {
    fn new() -> Self {
        Self {
            dirty_pages: vec![Vec::new(); CLIPMAP_LOD_COUNT],
            dirty_headers: Vec::new(),
            dirty_palette16_entries: Vec::new(),
            dirty_palette32_entries: Vec::new(),
            dirty_raw16_entries: Vec::new(),
        }
    }

    fn append_from(&mut self, dirty: &ClipmapDirtyState) {
        for lod in 0..CLIPMAP_LOD_COUNT {
            if let Some(src) = dirty.dirty_pages.get(lod) {
                self.dirty_pages[lod].extend_from_slice(src);
            }
        }
        self.dirty_headers.extend_from_slice(&dirty.dirty_headers);
        self.dirty_palette16_entries
            .extend_from_slice(&dirty.dirty_palette16_entries);
        self.dirty_palette32_entries
            .extend_from_slice(&dirty.dirty_palette32_entries);
        self.dirty_raw16_entries
            .extend_from_slice(&dirty.dirty_raw16_entries);
    }
}

/// GPU resources for clipmap rendering.
pub struct ClipmapRenderer {
    frame_buffers: Vec<FrameBuffers>,
    pending_dirty_per_frame: Vec<PendingDirtyState>,
    clipmap_info_addresses: Vec<vk::DeviceAddress>,
    deferred_deletions: DeferredDeletionQueue,
    /// Track target buffer sizes to sync across all frame buffers.
    /// When one frame grows a buffer, other frames will match that size.
    pool_buffer_sizes: PoolBufferSizes,
}

/// Tracks the target buffer sizes for pool buffers across all frames.
/// When one frame grows a buffer, we record the new size so other frames
/// can allocate to match without triggering repeated reallocations.
#[derive(Default)]
struct PoolBufferSizes {
    palette32: u64,
    raw16: u64,
}

impl ClipmapRenderer {
    /// Create a new clipmap renderer.
    pub fn new(frames_in_flight: usize) -> Self {
        Self {
            frame_buffers: (0..frames_in_flight).map(|_| FrameBuffers::new()).collect(),
            pending_dirty_per_frame: (0..frames_in_flight)
                .map(|_| PendingDirtyState::new())
                .collect(),
            clipmap_info_addresses: vec![0; frames_in_flight],
            deferred_deletions: DeferredDeletionQueue::new(frames_in_flight),
            pool_buffer_sizes: PoolBufferSizes::default(),
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
        _frame_number: u64,
    ) -> Result<()> {
        self.broadcast_dirty(&dirty);

        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.ensure_page_buffers").entered();
            self.ensure_page_buffers(allocator, frame_index, controller.active_lod_count())?;
        }
        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.ensure_info_buffers").entered();
            self.ensure_info_buffer(allocator, device, frame_index)?;
        }

        let store = controller.store();
        let pending = self.take_pending_dirty(frame_index);

        let reserve = controller.pool_reserve();

        let header_realloc = {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.ensure_brick_header_buffer").entered();
            self.ensure_brick_header_buffer(
                allocator,
                frame_index,
                store.headers(),
                reserve.headers as u64,
            )?
        };
        let pal16_full_upload_chunks = {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.ensure_palette16_buffer").entered();
            Self::ensure_palette16_pool(
                allocator,
                device,
                &mut self.frame_buffers[frame_index].palette16_pool,
                store.palette16_pool().len() as u64,
                reserve.palette16 as u64,
                &format!("clipmap_palette16_f{frame_index}"),
            )?
        };
        let pal32_realloc = {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.ensure_palette32_buffer").entered();
            let (full_upload, old_buffer, new_size) = Self::ensure_pool_buffer(
                allocator,
                &mut self.frame_buffers[frame_index].palette32_buffer,
                store.palette32_pool().len() as u64,
                PALETTE32_STRIDE as u64,
                reserve.palette32 as u64,
                self.pool_buffer_sizes.palette32,
                "clipmap_palette32",
            )?;
            self.pool_buffer_sizes.palette32 = self.pool_buffer_sizes.palette32.max(new_size);
            if let Some(mut buffer) = old_buffer {
                allocator.free_buffer(&mut buffer)?;
            }
            full_upload
        };
        let raw_realloc = {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.ensure_raw16_buffer").entered();
            let (full_upload, old_buffer, new_size) = Self::ensure_pool_buffer(
                allocator,
                &mut self.frame_buffers[frame_index].raw16_buffer,
                store.raw16_pool().len() as u64,
                RAW16_STRIDE as u64,
                reserve.raw16 as u64,
                self.pool_buffer_sizes.raw16,
                "clipmap_raw16",
            )?;
            self.pool_buffer_sizes.raw16 = self.pool_buffer_sizes.raw16.max(new_size);
            if let Some(mut buffer) = old_buffer {
                allocator.free_buffer(&mut buffer)?;
            }
            full_upload
        };

        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.upload_page_tables").entered();
            self.upload_page_tables(controller, frame_index, pending.dirty_pages)?;
        }
        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.upload_brick_headers").entered();
            self.upload_brick_headers(store, frame_index, pending.dirty_headers, header_realloc)?;
        }
        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.upload_palette16_entries").entered();
            self.upload_palette16_entries(
                store.palette16_pool(),
                &self.frame_buffers[frame_index].palette16_pool,
                pending.dirty_palette16_entries,
                pal16_full_upload_chunks,
            )?;
        }
        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.upload_palette32_entries").entered();
            self.upload_pool_entries(
                store.palette32_pool(),
                PALETTE32_STRIDE,
                self.frame_buffers[frame_index]
                    .palette32_buffer
                    .as_ref()
                    .unwrap(),
                pending.dirty_palette32_entries,
                pal32_realloc,
            )?;
        }
        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.upload_raw16_entries").entered();
            self.upload_pool_entries(
                store.raw16_pool(),
                RAW16_STRIDE,
                self.frame_buffers[frame_index]
                    .raw16_buffer
                    .as_ref()
                    .unwrap(),
                pending.dirty_raw16_entries,
                raw_realloc,
            )?;
        }

        let info = {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.build_gpu_info").entered();
            self.build_gpu_info(device, controller, frame_index)
        };
        if let Some(info_buffer) = &self.frame_buffers[frame_index].clipmap_info_buffer {
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
        self.deferred_deletions.process(allocator, frame_number)?;
        Ok(())
    }

    /// Destroy all GPU resources.
    pub fn destroy(mut self, allocator: &mut GpuAllocator) -> Result<()> {
        self.deferred_deletions.flush(allocator)?;

        for frame in &mut self.frame_buffers {
            for buffer in &mut frame.page_brick_buffers {
                if let Some(mut buf) = buffer.take() {
                    allocator.free_buffer(&mut buf)?;
                }
            }
            for buffer in &mut frame.page_occ_buffers {
                if let Some(mut buf) = buffer.take() {
                    allocator.free_buffer(&mut buf)?;
                }
            }
            for buffer in &mut frame.page_coord_buffers {
                if let Some(mut buf) = buffer.take() {
                    allocator.free_buffer(&mut buf)?;
                }
            }
            if let Some(mut buf) = frame.clipmap_info_buffer.take() {
                allocator.free_buffer(&mut buf)?;
            }
            if let Some(mut buf) = frame.brick_header_buffer.take() {
                allocator.free_buffer(&mut buf)?;
            }
            Self::destroy_palette16_pool(allocator, &mut frame.palette16_pool)?;
            if let Some(mut buf) = frame.palette32_buffer.take() {
                allocator.free_buffer(&mut buf)?;
            }
            if let Some(mut buf) = frame.raw16_buffer.take() {
                allocator.free_buffer(&mut buf)?;
            }
        }

        Ok(())
    }

    fn broadcast_dirty(&mut self, dirty: &ClipmapDirtyState) {
        for pending in &mut self.pending_dirty_per_frame {
            pending.append_from(dirty);
        }
    }

    fn take_pending_dirty(&mut self, frame_index: usize) -> PendingDirtyState {
        std::mem::replace(
            &mut self.pending_dirty_per_frame[frame_index],
            PendingDirtyState::new(),
        )
    }

    fn ensure_page_buffers(
        &mut self,
        allocator: &mut GpuAllocator,
        frame_index: usize,
        active_lod_count: usize,
    ) -> Result<()> {
        let page_count = CLIPMAP_PAGE_GRID * CLIPMAP_PAGE_GRID * CLIPMAP_PAGE_GRID;
        let brick_u32_count = page_count * PAGE_BRICKS;
        let occ_u32_count = page_count * 2;
        let coord_count = page_count;
        let usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        let frame = &mut self.frame_buffers[frame_index];

        for lod in 0..active_lod_count {
            if frame.page_brick_buffers[lod].is_none() {
                let mut buffer = allocator.create_buffer(
                    (brick_u32_count * std::mem::size_of::<u32>()) as u64,
                    usage,
                    MemoryLocation::CpuToGpu,
                    &format!("clipmap_page_bricks_f{frame_index}_lod{lod}"),
                )?;
                Self::initialize_u32_buffer(&mut buffer, 0, brick_u32_count)?;
                frame.page_brick_buffers[lod] = Some(buffer);
            }
            if frame.page_occ_buffers[lod].is_none() {
                let mut buffer = allocator.create_buffer(
                    (occ_u32_count * std::mem::size_of::<u32>()) as u64,
                    usage,
                    MemoryLocation::CpuToGpu,
                    &format!("clipmap_page_occ_f{frame_index}_lod{lod}"),
                )?;
                Self::initialize_u32_buffer(&mut buffer, 0, occ_u32_count)?;
                frame.page_occ_buffers[lod] = Some(buffer);
            }
            if frame.page_coord_buffers[lod].is_none() {
                let mut buffer = allocator.create_buffer(
                    (coord_count * std::mem::size_of::<[i32; 4]>()) as u64,
                    usage,
                    MemoryLocation::CpuToGpu,
                    &format!("clipmap_page_coord_f{frame_index}_lod{lod}"),
                )?;
                Self::initialize_page_coord_buffer(&mut buffer, coord_count)?;
                frame.page_coord_buffers[lod] = Some(buffer);
            }
        }

        for lod in active_lod_count..CLIPMAP_LOD_COUNT {
            if let Some(mut buffer) = frame.page_brick_buffers[lod].take() {
                allocator.free_buffer(&mut buffer)?;
            }
            if let Some(mut buffer) = frame.page_occ_buffers[lod].take() {
                allocator.free_buffer(&mut buffer)?;
            }
            if let Some(mut buffer) = frame.page_coord_buffers[lod].take() {
                allocator.free_buffer(&mut buffer)?;
            }
        }

        Ok(())
    }

    fn ensure_info_buffer(
        &mut self,
        allocator: &mut GpuAllocator,
        device: &ash::Device,
        frame_index: usize,
    ) -> Result<()> {
        if self.frame_buffers[frame_index]
            .clipmap_info_buffer
            .is_none()
        {
            let buffer = allocator.create_buffer(
                GpuClipmapInfo::SIZE as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryLocation::CpuToGpu,
                &format!("clipmap_info_{frame_index}"),
            )?;
            self.clipmap_info_addresses[frame_index] = buffer.device_address(device);
            self.frame_buffers[frame_index].clipmap_info_buffer = Some(buffer);
        }

        Ok(())
    }

    fn ensure_brick_header_buffer(
        &mut self,
        allocator: &mut GpuAllocator,
        frame_index: usize,
        headers: &[BrickHeader],
        min_entries: u64,
    ) -> Result<bool> {
        let header_size = std::mem::size_of::<BrickHeader>() as u64;
        let required = (headers.len() as u64)
            .saturating_mul(header_size)
            .max(min_entries.saturating_mul(header_size));
        let usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let buffer = &mut self.frame_buffers[frame_index].brick_header_buffer;

        if buffer.as_ref().is_none_or(|b| b.size < required) {
            let current_size = buffer.as_ref().map_or(0, |b| b.size);
            if let Some(mut old) = buffer.take() {
                allocator.free_buffer(&mut old)?;
            }
            let min_size = header_size;
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
                &format!("clipmap_brick_headers_f{frame_index}"),
            )?;
            *buffer = Some(new_buffer);
            return Ok(true);
        }

        Ok(false)
    }

    /// Ensures pool buffer exists and is large enough.
    /// Returns (full_upload_needed, old_buffer_to_free, new_buffer_size).
    fn ensure_pool_buffer(
        allocator: &mut GpuAllocator,
        buffer: &mut Option<GpuBuffer>,
        pool_size: u64,
        stride: u64,
        min_entries: u64,
        target_size: u64,
        name: &str,
    ) -> Result<(bool, Option<GpuBuffer>, u64)> {
        // Use minimum size based on render distance to avoid reallocations during world load.
        // Also respect target_size which may be larger due to another frame's growth.
        let min_size = min_entries * stride;
        let required = pool_size.max(stride).max(min_size).max(target_size);
        let usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        if buffer.as_ref().is_none_or(|b| b.size < required) {
            let current_size = buffer.as_ref().map_or(0, |b| b.size);
            // Use larger growth factor to reduce future reallocations.
            let grow_size = if current_size == 0 {
                required
            } else {
                current_size
                    .saturating_mul(POOL_GROWTH_FACTOR)
                    .max(required)
            };

            if current_size == 0 {
                tracing::debug!(
                    buffer = name,
                    size_bytes = grow_size,
                    size_mb = grow_size as f64 / (1024.0 * 1024.0),
                    entries = grow_size / stride,
                    "pool buffer initial allocation"
                );
            } else {
                tracing::warn!(
                    buffer = name,
                    old_bytes = current_size,
                    new_bytes = grow_size,
                    old_mb = current_size as f64 / (1024.0 * 1024.0),
                    new_mb = grow_size as f64 / (1024.0 * 1024.0),
                    growth_factor = grow_size as f64 / current_size as f64,
                    "pool buffer reallocation (potential frame spike)"
                );
            }

            let new_buffer =
                allocator.create_buffer(grow_size, usage, MemoryLocation::CpuToGpu, name)?;

            // Always request full upload on reallocation - upload_pool_entries will handle
            // the contiguous write efficiently. This avoids a separate preservation copy.
            let old_buffer = buffer.take();
            *buffer = Some(new_buffer);
            return Ok((true, old_buffer, grow_size));
        }

        let current_size = buffer.as_ref().map_or(0, |b| b.size);
        Ok((false, None, current_size))
    }

    fn ensure_palette16_pool(
        allocator: &mut GpuAllocator,
        device: &ash::Device,
        pool: &mut Palette16GpuPool,
        pool_size: u64,
        min_entries: u64,
        name: &str,
    ) -> Result<Vec<usize>> {
        let required_entries = entries_for_pool_size(pool_size, PALETTE16_STRIDE as u64)
            .max(min_entries as usize)
            .max(1);
        let required_chunks = required_entries.div_ceil(PALETTE16_CHUNK_ENTRIES);
        let usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        let old_chunk_count = pool.chunks.len();
        while pool.chunks.len() < required_chunks {
            let chunk_index = pool.chunks.len();
            let chunk = allocator.create_buffer(
                PALETTE16_CHUNK_BYTES,
                usage,
                MemoryLocation::CpuToGpu,
                &format!("{name}_chunk{chunk_index}"),
            )?;
            pool.chunks.push(chunk);
        }

        let mut table_recreated = false;
        if pool.chunks.len() > pool.table_capacity {
            let new_capacity = pool
                .table_capacity
                .max(1)
                .saturating_mul(2)
                .max(pool.chunks.len());
            if let Some(mut table) = pool.address_table.take() {
                allocator.free_buffer(&mut table)?;
            }
            pool.address_table = Some(allocator.create_buffer(
                (new_capacity * std::mem::size_of::<u64>()) as u64,
                usage,
                MemoryLocation::CpuToGpu,
                &format!("{name}_table"),
            )?);
            pool.table_capacity = new_capacity;
            table_recreated = true;
        }

        if table_recreated || pool.chunks.len() != old_chunk_count {
            Self::write_palette16_address_table(device, pool)?;
        }

        Ok((old_chunk_count..pool.chunks.len()).collect())
    }

    fn write_palette16_address_table(device: &ash::Device, pool: &Palette16GpuPool) -> Result<()> {
        let Some(table) = &pool.address_table else {
            return Ok(());
        };
        let addresses: Vec<u64> = pool
            .chunks
            .iter()
            .map(|chunk| chunk.device_address(device))
            .collect();
        table.write(&addresses)
    }

    fn destroy_palette16_pool(
        allocator: &mut GpuAllocator,
        pool: &mut Palette16GpuPool,
    ) -> Result<()> {
        if let Some(mut table) = pool.address_table.take() {
            allocator.free_buffer(&mut table)?;
        }
        for mut chunk in pool.chunks.drain(..) {
            allocator.free_buffer(&mut chunk)?;
        }
        pool.table_capacity = 0;
        Ok(())
    }

    #[cfg_attr(
        feature = "profiling-tracy",
        tracing::instrument(level = "trace", skip_all)
    )]
    fn upload_page_tables(
        &self,
        controller: &ClipmapStreamingController,
        frame_index: usize,
        dirty_pages: Vec<Vec<usize>>,
    ) -> Result<()> {
        let page_count = CLIPMAP_PAGE_GRID * CLIPMAP_PAGE_GRID * CLIPMAP_PAGE_GRID;
        let frame = &self.frame_buffers[frame_index];

        for lod in 0..CLIPMAP_LOD_COUNT {
            let Some(brick_buffer) = &frame.page_brick_buffers[lod] else {
                continue;
            };
            let Some(occ_buffer) = &frame.page_occ_buffers[lod] else {
                continue;
            };
            let Some(coord_buffer) = &frame.page_coord_buffers[lod] else {
                continue;
            };

            let page_bricks = controller.page_brick_indices(lod);
            let page_occ = controller.page_occ(lod);
            let page_coords = controller.page_coords(lod);

            let Some(lod_dirty_pages) = dirty_pages.get(lod) else {
                continue;
            };
            if lod_dirty_pages.is_empty() {
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
        frame_index: usize,
        mut dirty_headers: Vec<BrickId>,
        full_upload: bool,
    ) -> Result<()> {
        let Some(header_buffer) = &self.frame_buffers[frame_index].brick_header_buffer else {
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
        dirty_headers.sort_unstable_by_key(|id| id.0);
        dirty_headers.dedup_by_key(|id| id.0);
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
        mut entries: Vec<u32>,
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

        entries.sort_unstable();
        entries.dedup();

        if entries.is_empty() {
            return Ok(());
        }

        #[cfg(feature = "profiling-tracy")]
        let _span = tracing::trace_span!(
            "upload_pool_entries_incremental",
            entries = entries.len() as u32,
            stride = stride as u32
        )
        .entered();

        // Coalesce contiguous entries into single writes to reduce GPU memory operation overhead.
        // For example, entries [5, 6, 7, 10, 11] become two writes: [5..8] and [10..12].
        let mut run_start = entries[0];
        let mut run_len = 1u32;

        for &entry in &entries[1..] {
            if entry == run_start + run_len {
                // Consecutive entry - extend the current run
                run_len += 1;
            } else {
                // Gap found - write the current run and start a new one
                let offset = run_start as usize * stride;
                let end = (run_start + run_len) as usize * stride;
                if end <= pool.len() {
                    buffer.write_bytes(offset as u64, &pool[offset..end])?;
                }
                run_start = entry;
                run_len = 1;
            }
        }

        // Write the final run
        let offset = run_start as usize * stride;
        let end = (run_start + run_len) as usize * stride;
        if end <= pool.len() {
            buffer.write_bytes(offset as u64, &pool[offset..end])?;
        }

        Ok(())
    }

    #[cfg_attr(
        feature = "profiling-tracy",
        tracing::instrument(level = "trace", skip_all)
    )]
    fn upload_palette16_entries(
        &self,
        pool: &[u8],
        gpu_pool: &Palette16GpuPool,
        mut entries: Vec<u32>,
        full_upload_chunks: Vec<usize>,
    ) -> Result<()> {
        if pool.is_empty() {
            return Ok(());
        }

        for chunk_index in full_upload_chunks {
            let Some(chunk) = gpu_pool.chunks.get(chunk_index) else {
                continue;
            };
            let start = chunk_index * PALETTE16_CHUNK_ENTRIES * PALETTE16_STRIDE;
            if start >= pool.len() {
                continue;
            }
            let end = (start + PALETTE16_CHUNK_ENTRIES * PALETTE16_STRIDE).min(pool.len());
            chunk.write_bytes(0, &pool[start..end])?;
        }

        entries.sort_unstable();
        entries.dedup();

        if entries.is_empty() {
            return Ok(());
        }

        #[cfg(feature = "profiling-tracy")]
        let _span = tracing::trace_span!(
            "upload_palette16_entries_incremental",
            entries = entries.len() as u32
        )
        .entered();

        let mut run_start = entries[0];
        let mut run_len = 1u32;

        for &entry in &entries[1..] {
            let current_chunk = entry as usize / PALETTE16_CHUNK_ENTRIES;
            let run_chunk = run_start as usize / PALETTE16_CHUNK_ENTRIES;
            if entry == run_start + run_len && current_chunk == run_chunk {
                run_len += 1;
            } else {
                Self::write_palette16_run(pool, gpu_pool, run_start, run_len)?;
                run_start = entry;
                run_len = 1;
            }
        }

        Self::write_palette16_run(pool, gpu_pool, run_start, run_len)
    }

    fn write_palette16_run(
        pool: &[u8],
        gpu_pool: &Palette16GpuPool,
        run_start: u32,
        run_len: u32,
    ) -> Result<()> {
        let chunk_index = run_start as usize / PALETTE16_CHUNK_ENTRIES;
        let local_start = run_start as usize % PALETTE16_CHUNK_ENTRIES;
        let Some(chunk) = gpu_pool.chunks.get(chunk_index) else {
            return Ok(());
        };

        let src_offset = run_start as usize * PALETTE16_STRIDE;
        let src_end = (run_start + run_len) as usize * PALETTE16_STRIDE;
        if src_end > pool.len() {
            return Ok(());
        }
        let dst_offset = (local_start * PALETTE16_STRIDE) as u64;
        chunk.write_bytes(dst_offset, &pool[src_offset..src_end])
    }

    #[cfg_attr(
        feature = "profiling-tracy",
        tracing::instrument(level = "trace", skip_all)
    )]
    fn build_gpu_info(
        &self,
        device: &ash::Device,
        controller: &ClipmapStreamingController,
        frame_index: usize,
    ) -> GpuClipmapInfo {
        let mut info = GpuClipmapInfo::zeroed();
        let frame = &self.frame_buffers[frame_index];
        let render_origin = controller.camera_voxel();

        for lod in 0..CLIPMAP_LOD_COUNT {
            if let Some(buffer) = &frame.page_brick_buffers[lod] {
                info.page_brick_indices_addr[lod] = buffer.device_address(device);
            }
            if let Some(buffer) = &frame.page_occ_buffers[lod] {
                info.page_occ_addr[lod] = buffer.device_address(device);
            }
            if let Some(buffer) = &frame.page_coord_buffers[lod] {
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
            let page_size = i64::from(voxel_size.max(1)) * PAGE_VOXELS_PER_AXIS as i64;
            let render_page_origin = [
                div_floor(render_origin.x, page_size),
                div_floor(render_origin.y, page_size),
                div_floor(render_origin.z, page_size),
            ];
            let render_page_offset = [
                render_origin.x - render_page_origin[0] * page_size,
                render_origin.y - render_page_origin[1] * page_size,
                render_origin.z - render_page_origin[2] * page_size,
            ];
            let lod_min = [
                (origin.x - render_origin.x) as f32,
                (origin.y - render_origin.y) as f32,
                (origin.z - render_origin.z) as f32,
            ];

            info.origin[lod] = [
                render_page_origin[0] as i32,
                render_page_origin[1] as i32,
                render_page_origin[2] as i32,
                0,
            ];
            info.voxel_size[lod] = [
                voxel_size,
                render_page_offset[0] as u32,
                render_page_offset[1] as u32,
                render_page_offset[2] as u32,
            ];
            info.lod_aabb_min[lod] = [lod_min[0], lod_min[1], lod_min[2], 0.0];
            info.lod_aabb_max[lod] = [
                lod_min[0] + coverage,
                lod_min[1] + coverage,
                lod_min[2] + coverage,
                0.0,
            ];
        }

        if let Some(buffer) = &frame.brick_header_buffer {
            info.brick_header_addr = buffer.device_address(device);
        }
        if let Some(buffer) = &frame.palette16_pool.address_table {
            info.palette16_addr = buffer.device_address(device);
        }
        if let Some(buffer) = &frame.palette32_buffer {
            info.palette32_addr = buffer.device_address(device);
        }
        if let Some(buffer) = &frame.raw16_buffer {
            info.raw16_addr = buffer.device_address(device);
        }

        info
    }

    fn initialize_u32_buffer(buffer: &mut GpuBuffer, value: u32, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }

        let chunk = vec![value; INIT_CHUNK_U32];
        let mut offset_elems = 0usize;
        while offset_elems < count {
            let len = (count - offset_elems).min(chunk.len());
            let offset_bytes = (offset_elems * std::mem::size_of::<u32>()) as u64;
            buffer.write_range(offset_bytes, &chunk[..len])?;
            offset_elems += len;
        }

        Ok(())
    }

    fn initialize_page_coord_buffer(buffer: &mut GpuBuffer, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }

        let chunk = vec![INVALID_PAGE_COORD; INIT_CHUNK_COORD];
        let mut offset_elems = 0usize;
        while offset_elems < count {
            let len = (count - offset_elems).min(chunk.len());
            let offset_bytes = (offset_elems * std::mem::size_of::<[i32; 4]>()) as u64;
            buffer.write_range(offset_bytes, &chunk[..len])?;
            offset_elems += len;
        }

        Ok(())
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
