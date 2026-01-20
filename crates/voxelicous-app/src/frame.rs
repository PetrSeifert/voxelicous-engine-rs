//! Per-frame context for rendering.

use ash::vk;

/// Context for the current frame being rendered.
///
/// Provides access to the command buffer and swapchain image for this frame.
pub struct FrameContext {
    /// Command buffer for recording rendering commands.
    pub command_buffer: vk::CommandBuffer,
    /// Index of the acquired swapchain image.
    pub image_index: u32,
    /// The swapchain image for this frame.
    pub swapchain_image: vk::Image,
    /// Delta time since last frame in seconds.
    pub dt: f32,
    /// Current frame number.
    pub frame_number: u64,
}

impl FrameContext {
    /// Create a new frame context.
    pub(crate) fn new(
        command_buffer: vk::CommandBuffer,
        image_index: u32,
        swapchain_image: vk::Image,
        dt: f32,
        frame_number: u64,
    ) -> Self {
        Self {
            command_buffer,
            image_index,
            swapchain_image,
            dt,
            frame_number,
        }
    }
}
