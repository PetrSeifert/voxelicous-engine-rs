//! Async chunk upload manager for non-blocking GPU transfers.
//!
//! Uses timeline semaphores and a dedicated transfer queue to upload
//! chunk data asynchronously, reducing frame stalls from blocking uploads.

use ash::vk;
use gpu_allocator::MemoryLocation;
use voxelicous_core::coords::ChunkPos;
use voxelicous_gpu::error::Result;
use voxelicous_gpu::memory::{GpuAllocator, GpuBuffer};
use voxelicous_gpu::TimelineSemaphore;
use voxelicous_voxel::{GpuOctreeNode, SvoDag, VoxelStorage};

/// Number of staging slots in the ring buffer.
const NUM_STAGING_SLOTS: usize = 3;

/// Size of each staging buffer slot in bytes (2MB).
const STAGING_SLOT_SIZE: u64 = 2 * 1024 * 1024;

/// Information about a pending chunk upload.
#[derive(Clone)]
pub struct PendingChunkUpload {
    /// Chunk position in world coordinates.
    pub pos: ChunkPos,
    /// Offset into the staging buffer.
    pub staging_offset: u64,
    /// Size of the data in bytes.
    pub data_size: u64,
    /// Root node index of the SVO-DAG.
    pub root_index: u32,
    /// Depth of the SVO-DAG.
    pub depth: u32,
}

/// Information about a completed chunk upload.
pub struct CompletedUpload {
    /// Chunk position.
    pub pos: ChunkPos,
    /// The GPU buffer containing the uploaded data.
    pub node_buffer: GpuBuffer,
    /// Root node index.
    pub root_index: u32,
    /// SVO-DAG depth.
    pub depth: u32,
    /// Device address of the buffer.
    pub device_address: vk::DeviceAddress,
}

/// A staging slot in the ring buffer.
struct UploadSlot {
    /// Staging buffer (CPU-visible).
    staging_buffer: GpuBuffer,
    /// Current write offset into the staging buffer.
    write_offset: u64,
    /// Timeline value when this slot's upload will be complete.
    timeline_value: u64,
    /// Chunks pending upload in this slot.
    pending_chunks: Vec<PendingChunkUpload>,
    /// GPU buffers being populated by this slot's transfer.
    gpu_buffers: Vec<(ChunkPos, GpuBuffer, u32, u32)>, // (pos, buffer, root_index, depth)
    /// Whether this slot has pending work to submit.
    has_pending_work: bool,
}

impl UploadSlot {
    fn new(staging_buffer: GpuBuffer) -> Self {
        Self {
            staging_buffer,
            write_offset: 0,
            timeline_value: 0,
            pending_chunks: Vec::new(),
            gpu_buffers: Vec::new(),
            has_pending_work: false,
        }
    }

    fn reset(&mut self) {
        self.write_offset = 0;
        self.pending_chunks.clear();
        self.gpu_buffers.clear();
        self.has_pending_work = false;
    }

    fn available_space(&self) -> u64 {
        self.staging_buffer.size.saturating_sub(self.write_offset)
    }
}

/// Manages async chunk uploads using a ring buffer of staging slots.
pub struct AsyncUploadManager {
    /// Ring buffer of staging slots.
    staging_slots: Vec<UploadSlot>,
    /// Current slot being filled.
    current_slot: usize,
    /// Timeline semaphore for upload synchronization.
    timeline: TimelineSemaphore,
    /// Command pool for transfer operations.
    transfer_pool: vk::CommandPool,
    /// Command buffers for each slot.
    transfer_buffers: Vec<vk::CommandBuffer>,
    /// Transfer queue family index.
    transfer_queue_family: u32,
    /// Last submitted timeline value.
    last_submitted_value: u64,
}

impl AsyncUploadManager {
    /// Create a new async upload manager.
    ///
    /// # Arguments
    /// * `device` - Vulkan device handle.
    /// * `allocator` - GPU memory allocator.
    /// * `transfer_queue_family` - Queue family index for transfer operations.
    ///
    /// # Safety
    /// The device must be valid.
    pub unsafe fn new(
        device: &ash::Device,
        allocator: &mut GpuAllocator,
        transfer_queue_family: u32,
    ) -> Result<Self> {
        // Create timeline semaphore
        let timeline = TimelineSemaphore::new(device)?;

        // Create command pool for transfer queue
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(transfer_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let transfer_pool = device.create_command_pool(&pool_info, None)?;

        // Allocate command buffers for each slot
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(transfer_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(NUM_STAGING_SLOTS as u32);

        let transfer_buffers = device.allocate_command_buffers(&alloc_info)?;

        // Create staging slots
        let mut staging_slots = Vec::with_capacity(NUM_STAGING_SLOTS);
        for i in 0..NUM_STAGING_SLOTS {
            let staging_buffer = allocator.create_buffer(
                STAGING_SLOT_SIZE,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                &format!("async_upload_staging_{i}"),
            )?;
            staging_slots.push(UploadSlot::new(staging_buffer));
        }

        Ok(Self {
            staging_slots,
            current_slot: 0,
            timeline,
            transfer_pool,
            transfer_buffers,
            transfer_queue_family,
            last_submitted_value: 0,
        })
    }

    /// Get the timeline semaphore handle for synchronization.
    pub fn timeline_semaphore(&self) -> vk::Semaphore {
        self.timeline.handle()
    }

    /// Get the last submitted timeline value.
    pub fn last_submitted_value(&self) -> u64 {
        self.last_submitted_value
    }

    /// Get the transfer queue family index.
    pub fn transfer_queue_family(&self) -> u32 {
        self.transfer_queue_family
    }

    /// Queue a chunk for async upload.
    ///
    /// Returns `true` if the chunk was queued, `false` if there's no space.
    /// Call `submit_pending` then `poll_completed` to make room.
    pub fn queue_chunk(
        &mut self,
        allocator: &mut GpuAllocator,
        _device: &ash::Device,
        pos: ChunkPos,
        dag: &SvoDag,
    ) -> Result<bool> {
        // Convert DAG to GPU buffer format
        let nodes = dag.to_gpu_buffer();
        if nodes.is_empty() {
            return Ok(false);
        }

        let data_size = (nodes.len() * std::mem::size_of::<GpuOctreeNode>()) as u64;

        // Find a slot with enough space
        let slot = &mut self.staging_slots[self.current_slot];

        // If current slot doesn't have space, return false
        if slot.available_space() < data_size {
            return Ok(false);
        }

        // Write data to staging buffer
        let staging_offset = slot.write_offset;
        let staging_ptr = slot.staging_buffer.mapped_ptr().ok_or_else(|| {
            voxelicous_gpu::error::GpuError::InvalidState("Staging buffer not mapped".to_string())
        })?;

        unsafe {
            let dst = staging_ptr.add(staging_offset as usize);
            std::ptr::copy_nonoverlapping(nodes.as_ptr() as *const u8, dst, data_size as usize);
        }

        // Create destination GPU buffer
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST;

        let gpu_buffer =
            allocator.create_buffer(data_size, usage, MemoryLocation::GpuOnly, "svo_dag_async")?;

        // Record pending upload
        slot.pending_chunks.push(PendingChunkUpload {
            pos,
            staging_offset,
            data_size,
            root_index: dag.root(),
            depth: dag.depth(),
        });

        slot.gpu_buffers
            .push((pos, gpu_buffer, dag.root(), dag.depth()));
        slot.write_offset += data_size;
        slot.has_pending_work = true;

        Ok(true)
    }

    /// Submit pending uploads to the transfer queue.
    ///
    /// Returns the timeline value that will be signaled when uploads complete.
    ///
    /// # Safety
    /// The device and queue must be valid.
    pub unsafe fn submit_pending(
        &mut self,
        device: &ash::Device,
        transfer_queue: vk::Queue,
    ) -> Result<u64> {
        let slot = &mut self.staging_slots[self.current_slot];

        if !slot.has_pending_work || slot.pending_chunks.is_empty() {
            return Ok(self.last_submitted_value);
        }

        let cmd = self.transfer_buffers[self.current_slot];

        // Reset and begin command buffer
        device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device.begin_command_buffer(cmd, &begin_info)?;

        // Record copy commands for each pending chunk
        for (i, pending) in slot.pending_chunks.iter().enumerate() {
            let (_, gpu_buffer, _, _) = &slot.gpu_buffers[i];

            let copy_region = vk::BufferCopy {
                src_offset: pending.staging_offset,
                dst_offset: 0,
                size: pending.data_size,
            };

            device.cmd_copy_buffer(
                cmd,
                slot.staging_buffer.buffer,
                gpu_buffer.buffer,
                &[copy_region],
            );
        }

        device.end_command_buffer(cmd)?;

        // Get timeline value for this submission
        let signal_value = self.timeline.next_signal_value();
        slot.timeline_value = signal_value;

        // Submit with timeline semaphore signal
        let signal_semaphores = [self.timeline.handle()];
        let signal_values = [signal_value];

        let mut timeline_info =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);

        let command_buffers = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut timeline_info);

        device.queue_submit(transfer_queue, &[submit_info], vk::Fence::null())?;

        self.last_submitted_value = signal_value;

        // Advance to next slot
        self.current_slot = (self.current_slot + 1) % NUM_STAGING_SLOTS;

        Ok(signal_value)
    }

    /// Poll for completed uploads (non-blocking).
    ///
    /// Returns a vector of completed uploads that can now be used.
    ///
    /// # Safety
    /// The device must be valid.
    pub unsafe fn poll_completed(&mut self, device: &ash::Device) -> Result<Vec<CompletedUpload>> {
        let mut completed = Vec::new();

        for slot in &mut self.staging_slots {
            if slot.timeline_value == 0 || slot.gpu_buffers.is_empty() {
                continue;
            }

            // Check if this slot's transfer is complete
            if self.timeline.check_completed(device, slot.timeline_value)? {
                // Collect completed uploads
                for (pos, buffer, root_index, depth) in slot.gpu_buffers.drain(..) {
                    let device_address = buffer.device_address(device);
                    completed.push(CompletedUpload {
                        pos,
                        node_buffer: buffer,
                        root_index,
                        depth,
                        device_address,
                    });
                }

                // Reset the slot for reuse
                slot.reset();
            }
        }

        Ok(completed)
    }

    /// Get the number of chunks currently queued for upload.
    pub fn queued_count(&self) -> usize {
        self.staging_slots
            .iter()
            .map(|s| s.pending_chunks.len())
            .sum()
    }

    /// Check if there's space to queue more chunks.
    pub fn has_space(&self) -> bool {
        let slot = &self.staging_slots[self.current_slot];
        slot.available_space() > std::mem::size_of::<GpuOctreeNode>() as u64 * 1024
    }

    /// Wait for all pending uploads to complete.
    ///
    /// # Safety
    /// The device must be valid.
    pub unsafe fn flush(&mut self, device: &ash::Device) -> Result<()> {
        if self.last_submitted_value > 0 {
            self.timeline
                .wait(device, self.last_submitted_value, u64::MAX)?;
        }
        Ok(())
    }

    /// Destroy the upload manager and free all resources.
    ///
    /// # Safety
    /// The device must be valid and all uploads must be complete.
    pub unsafe fn destroy(
        mut self,
        device: &ash::Device,
        allocator: &mut GpuAllocator,
    ) -> Result<()> {
        // Wait for any pending uploads
        self.flush(device)?;

        // Free staging buffers
        for slot in &mut self.staging_slots {
            allocator.free_buffer(&mut slot.staging_buffer)?;
            // Free any remaining GPU buffers
            for (_, mut buffer, _, _) in slot.gpu_buffers.drain(..) {
                allocator.free_buffer(&mut buffer)?;
            }
        }

        // Destroy command pool (frees command buffers automatically)
        device.destroy_command_pool(self.transfer_pool, None);

        // Destroy timeline semaphore
        self.timeline.destroy(device);

        Ok(())
    }
}
