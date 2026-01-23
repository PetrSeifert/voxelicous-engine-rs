//! Synchronization primitives.

use crate::error::Result;
use ash::vk;
use std::sync::atomic::{AtomicU64, Ordering};

/// Create a semaphore.
///
/// # Safety
/// The device must be valid.
pub unsafe fn create_semaphore(device: &ash::Device) -> Result<vk::Semaphore> {
    let create_info = vk::SemaphoreCreateInfo::default();
    let semaphore = device.create_semaphore(&create_info, None)?;
    Ok(semaphore)
}

/// Create a fence.
///
/// # Safety
/// The device must be valid.
pub unsafe fn create_fence(device: &ash::Device, signaled: bool) -> Result<vk::Fence> {
    let flags = if signaled {
        vk::FenceCreateFlags::SIGNALED
    } else {
        vk::FenceCreateFlags::empty()
    };

    let create_info = vk::FenceCreateInfo::default().flags(flags);
    let fence = device.create_fence(&create_info, None)?;
    Ok(fence)
}

/// Wait for a fence to be signaled.
///
/// # Safety
/// The device and fence must be valid.
pub unsafe fn wait_for_fence(
    device: &ash::Device,
    fence: vk::Fence,
    timeout_ns: u64,
) -> Result<()> {
    device.wait_for_fences(&[fence], true, timeout_ns)?;
    Ok(())
}

/// Reset a fence to unsignaled state.
///
/// # Safety
/// The device and fence must be valid.
pub unsafe fn reset_fence(device: &ash::Device, fence: vk::Fence) -> Result<()> {
    device.reset_fences(&[fence])?;
    Ok(())
}

/// Frame synchronization resources.
pub struct FrameSync {
    /// Semaphore signaled when image is available
    pub image_available: vk::Semaphore,
    /// Semaphore signaled when rendering is complete
    pub render_finished: vk::Semaphore,
    /// Fence to wait for frame completion
    pub in_flight: vk::Fence,
}

impl FrameSync {
    /// Create frame synchronization resources.
    ///
    /// # Safety
    /// The device must be valid.
    pub unsafe fn new(device: &ash::Device) -> Result<Self> {
        Ok(Self {
            image_available: create_semaphore(device)?,
            render_finished: create_semaphore(device)?,
            in_flight: create_fence(device, true)?,
        })
    }

    /// Wait for this frame to be available.
    ///
    /// # Safety
    /// The device must be valid.
    pub unsafe fn wait(&self, device: &ash::Device) -> Result<()> {
        wait_for_fence(device, self.in_flight, u64::MAX)
    }

    /// Reset the fence for the next frame.
    ///
    /// # Safety
    /// The device must be valid.
    pub unsafe fn reset(&self, device: &ash::Device) -> Result<()> {
        reset_fence(device, self.in_flight)
    }

    /// Destroy synchronization resources.
    ///
    /// # Safety
    /// The device must be valid and resources must not be in use.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_semaphore(self.image_available, None);
        device.destroy_semaphore(self.render_finished, None);
        device.destroy_fence(self.in_flight, None);
    }
}

/// Manages synchronization for multiple frames in flight.
pub struct FrameSyncManager {
    frame_syncs: Vec<FrameSync>,
    current_frame: usize,
}

impl FrameSyncManager {
    /// Create a sync manager for the given number of frames in flight.
    ///
    /// # Safety
    /// The device must be valid.
    pub unsafe fn new(device: &ash::Device, frames_in_flight: usize) -> Result<Self> {
        let mut frame_syncs = Vec::with_capacity(frames_in_flight);
        for _ in 0..frames_in_flight {
            frame_syncs.push(FrameSync::new(device)?);
        }

        Ok(Self {
            frame_syncs,
            current_frame: 0,
        })
    }

    /// Get the current frame's sync resources.
    pub fn current(&self) -> &FrameSync {
        &self.frame_syncs[self.current_frame]
    }

    /// Advance to the next frame.
    pub fn advance(&mut self) {
        self.current_frame = (self.current_frame + 1) % self.frame_syncs.len();
    }

    /// Get the current frame index.
    pub fn current_frame(&self) -> usize {
        self.current_frame
    }

    /// Destroy all resources.
    ///
    /// # Safety
    /// The device must be valid and all resources must not be in use.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        for sync in &self.frame_syncs {
            sync.destroy(device);
        }
    }
}

/// Timeline semaphore for GPU-GPU synchronization.
///
/// Timeline semaphores allow for more flexible synchronization than binary semaphores.
/// They maintain an incrementing counter and can be waited on or signaled with specific values.
/// This enables async upload patterns where transfers can complete in any order.
pub struct TimelineSemaphore {
    semaphore: vk::Semaphore,
    current_value: AtomicU64,
}

impl TimelineSemaphore {
    /// Create a new timeline semaphore with initial value 0.
    ///
    /// # Safety
    /// The device must be valid.
    pub unsafe fn new(device: &ash::Device) -> Result<Self> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);

        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);

        let semaphore = device.create_semaphore(&create_info, None)?;

        Ok(Self {
            semaphore,
            current_value: AtomicU64::new(0),
        })
    }

    /// Get the raw semaphore handle.
    pub fn handle(&self) -> vk::Semaphore {
        self.semaphore
    }

    /// Get the next value to signal and increment the internal counter.
    ///
    /// Call this to get the value to use when signaling this semaphore.
    pub fn next_signal_value(&self) -> u64 {
        self.current_value.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Get the current counter value (last signaled value).
    pub fn current_value(&self) -> u64 {
        self.current_value.load(Ordering::SeqCst)
    }

    /// Check if a specific value has been completed (non-blocking).
    ///
    /// Returns `true` if the semaphore has reached at least the given value.
    ///
    /// # Safety
    /// The device must be valid.
    pub unsafe fn check_completed(&self, device: &ash::Device, value: u64) -> Result<bool> {
        let counter_value = self.query_value(device)?;
        Ok(counter_value >= value)
    }

    /// Query the current GPU-side semaphore counter value.
    ///
    /// # Safety
    /// The device must be valid.
    pub unsafe fn query_value(&self, device: &ash::Device) -> Result<u64> {
        let value = device.get_semaphore_counter_value(self.semaphore)?;
        Ok(value)
    }

    /// Wait for the semaphore to reach a specific value.
    ///
    /// # Arguments
    /// * `device` - Vulkan device handle.
    /// * `value` - Value to wait for.
    /// * `timeout_ns` - Timeout in nanoseconds (u64::MAX for infinite).
    ///
    /// # Safety
    /// The device must be valid.
    pub unsafe fn wait(&self, device: &ash::Device, value: u64, timeout_ns: u64) -> Result<()> {
        let semaphores = [self.semaphore];
        let values = [value];

        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);

        device.wait_semaphores(&wait_info, timeout_ns)?;
        Ok(())
    }

    /// Signal the semaphore to a specific value from the host.
    ///
    /// # Safety
    /// The device must be valid. The value must be greater than the current value.
    pub unsafe fn signal(&self, device: &ash::Device, value: u64) -> Result<()> {
        let signal_info = vk::SemaphoreSignalInfo::default()
            .semaphore(self.semaphore)
            .value(value);

        device.signal_semaphore(&signal_info)?;
        Ok(())
    }

    /// Destroy the semaphore.
    ///
    /// # Safety
    /// The device must be valid and the semaphore must not be in use.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_semaphore(self.semaphore, None);
    }
}
