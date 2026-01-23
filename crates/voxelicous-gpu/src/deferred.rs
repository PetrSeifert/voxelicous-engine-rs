//! Deferred resource deletion for multi-frame-in-flight rendering.
//!
//! When using multiple frames in flight, GPU resources cannot be freed immediately
//! as they may still be in use by a previous frame. This module provides a queue
//! to defer deletions until the resource is guaranteed to no longer be in use.

use crate::error::Result;
use crate::memory::{GpuAllocator, GpuBuffer};

/// A buffer pending deletion.
pub struct PendingDeletion {
    /// The buffer to be freed.
    pub buffer: GpuBuffer,
    /// Frame number when this buffer was queued for deletion.
    pub frame_queued: u64,
}

/// Queue for deferred buffer deletions.
///
/// Buffers are queued with a frame number and only freed once enough frames
/// have passed to guarantee they are no longer in use by any in-flight frame.
pub struct DeferredDeletionQueue {
    /// Buffers pending deletion.
    pending: Vec<PendingDeletion>,
    /// Number of frames in flight (determines how long to wait before freeing).
    frames_in_flight: usize,
}

impl DeferredDeletionQueue {
    /// Create a new deferred deletion queue.
    ///
    /// # Arguments
    /// * `frames_in_flight` - Number of frames that can be in flight simultaneously.
    ///   Buffers will be kept for this many frames before being freed.
    pub fn new(frames_in_flight: usize) -> Self {
        Self {
            pending: Vec::new(),
            frames_in_flight,
        }
    }

    /// Queue a buffer for deferred deletion.
    ///
    /// The buffer will be freed once `frames_in_flight` frames have passed.
    ///
    /// # Arguments
    /// * `buffer` - The buffer to queue for deletion.
    /// * `frame_number` - Current frame number when queuing.
    pub fn queue(&mut self, buffer: GpuBuffer, frame_number: u64) {
        self.pending.push(PendingDeletion {
            buffer,
            frame_queued: frame_number,
        });
    }

    /// Process the queue and free buffers that are safe to delete.
    ///
    /// Call this at the start of each frame to free resources from completed frames.
    ///
    /// # Arguments
    /// * `allocator` - GPU allocator for freeing buffers.
    /// * `current_frame_number` - Current frame number.
    pub fn process(&mut self, allocator: &mut GpuAllocator, current_frame_number: u64) -> Result<()> {
        // Keep buffers that were queued within the last `frames_in_flight` frames
        let cutoff = current_frame_number.saturating_sub(self.frames_in_flight as u64);

        // Collect items to free
        let mut to_free = Vec::new();
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].frame_queued < cutoff {
                to_free.push(self.pending.remove(i));
            } else {
                i += 1;
            }
        }

        // Actually free the buffers
        for mut pending in to_free {
            allocator.free_buffer(&mut pending.buffer)?;
        }

        Ok(())
    }

    /// Flush all pending deletions immediately.
    ///
    /// Call this during shutdown after `device_wait_idle()` to ensure
    /// all resources are freed.
    ///
    /// # Arguments
    /// * `allocator` - GPU allocator for freeing buffers.
    pub fn flush(&mut self, allocator: &mut GpuAllocator) -> Result<()> {
        for mut pending in self.pending.drain(..) {
            allocator.free_buffer(&mut pending.buffer)?;
        }
        Ok(())
    }

    /// Get the number of pending deletions.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Update the frames in flight count.
    ///
    /// Call this when the swapchain is recreated with a different image count.
    pub fn set_frames_in_flight(&mut self, frames_in_flight: usize) {
        self.frames_in_flight = frames_in_flight;
    }
}
