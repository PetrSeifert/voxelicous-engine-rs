//! Synchronization primitives.

use crate::error::Result;
use ash::vk;

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
#[cfg_attr(feature = "profiling-tracy", tracing::instrument(level = "trace", skip_all))]
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
#[cfg_attr(feature = "profiling-tracy", tracing::instrument(level = "trace", skip_all))]
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
