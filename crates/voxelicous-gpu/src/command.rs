//! Command buffer management.

use crate::error::Result;
use ash::vk;

/// Command pool for allocating command buffers.
pub struct CommandPool {
    pool: vk::CommandPool,
    queue_family: u32,
}

impl CommandPool {
    /// Create a new command pool.
    ///
    /// # Safety
    /// The device must be valid and the queue family must exist.
    pub unsafe fn new(
        device: &ash::Device,
        queue_family: u32,
        flags: vk::CommandPoolCreateFlags,
    ) -> Result<Self> {
        let create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(flags);

        let pool = device.create_command_pool(&create_info, None)?;

        Ok(Self { pool, queue_family })
    }

    /// Get the raw pool handle.
    pub fn handle(&self) -> vk::CommandPool {
        self.pool
    }

    /// Get the queue family index.
    pub fn queue_family(&self) -> u32 {
        self.queue_family
    }

    /// Allocate a single command buffer.
    ///
    /// # Safety
    /// The device must be valid.
    pub unsafe fn allocate_command_buffer(
        &self,
        device: &ash::Device,
        level: vk::CommandBufferLevel,
    ) -> Result<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(level)
            .command_buffer_count(1);

        let buffers = device.allocate_command_buffers(&alloc_info)?;
        Ok(buffers[0])
    }

    /// Allocate multiple command buffers.
    ///
    /// # Safety
    /// The device must be valid.
    pub unsafe fn allocate_command_buffers(
        &self,
        device: &ash::Device,
        level: vk::CommandBufferLevel,
        count: u32,
    ) -> Result<Vec<vk::CommandBuffer>> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(level)
            .command_buffer_count(count);

        let buffers = device.allocate_command_buffers(&alloc_info)?;
        Ok(buffers)
    }

    /// Reset the command pool.
    ///
    /// # Safety
    /// The device must be valid and all command buffers from this pool must not be in use.
    pub unsafe fn reset(
        &self,
        device: &ash::Device,
        flags: vk::CommandPoolResetFlags,
    ) -> Result<()> {
        device.reset_command_pool(self.pool, flags)?;
        Ok(())
    }

    /// Destroy the command pool.
    ///
    /// # Safety
    /// The device must be valid and the pool must not be in use.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_command_pool(self.pool, None);
    }
}

/// Begin recording a command buffer.
///
/// # Safety
/// The device and command buffer must be valid.
pub unsafe fn begin_command_buffer(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    flags: vk::CommandBufferUsageFlags,
) -> Result<()> {
    let begin_info = vk::CommandBufferBeginInfo::default().flags(flags);
    device.begin_command_buffer(cmd, &begin_info)?;
    Ok(())
}

/// End recording a command buffer.
///
/// # Safety
/// The device and command buffer must be valid.
pub unsafe fn end_command_buffer(device: &ash::Device, cmd: vk::CommandBuffer) -> Result<()> {
    device.end_command_buffer(cmd)?;
    Ok(())
}

/// Submit command buffers to a queue.
///
/// # Safety
/// All handles must be valid.
#[cfg_attr(feature = "profiling-tracy", tracing::instrument(level = "trace", skip_all))]
pub unsafe fn submit_command_buffers(
    device: &ash::Device,
    queue: vk::Queue,
    command_buffers: &[vk::CommandBuffer],
    wait_semaphores: &[vk::Semaphore],
    wait_stages: &[vk::PipelineStageFlags],
    signal_semaphores: &[vk::Semaphore],
    fence: vk::Fence,
) -> Result<()> {
    let submit_info = vk::SubmitInfo::default()
        .command_buffers(command_buffers)
        .wait_semaphores(wait_semaphores)
        .wait_dst_stage_mask(wait_stages)
        .signal_semaphores(signal_semaphores);

    device.queue_submit(queue, &[submit_info], fence)?;
    Ok(())
}

/// Execute a single-time command buffer.
///
/// # Safety
/// All handles must be valid.
pub unsafe fn execute_single_time_commands<F>(
    device: &ash::Device,
    pool: &CommandPool,
    queue: vk::Queue,
    f: F,
) -> Result<()>
where
    F: FnOnce(vk::CommandBuffer),
{
    let cmd = pool.allocate_command_buffer(device, vk::CommandBufferLevel::PRIMARY)?;

    begin_command_buffer(device, cmd, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;
    f(cmd);
    end_command_buffer(device, cmd)?;

    let cmd_buffers = [cmd];
    let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_buffers);
    device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
    device.queue_wait_idle(queue)?;

    device.free_command_buffers(pool.handle(), &[cmd]);

    Ok(())
}
