//! `VoxelApp` trait definition.

use crate::context::AppContext;
use crate::frame::FrameContext;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};

/// Trait for Voxelicous applications.
///
/// Implement this trait to create a new application using the Voxelicous engine.
/// The framework handles all boilerplate like window creation, GPU initialization,
/// swapchain management, and event loop handling.
pub trait VoxelApp: Sized {
    /// Initialize the application.
    ///
    /// Called once when the application starts, after the GPU context and
    /// window have been created.
    fn init(ctx: &mut AppContext) -> anyhow::Result<Self>;

    /// Update application state.
    ///
    /// Called every frame before rendering. Use this to update animation,
    /// physics, or other time-dependent state.
    ///
    /// # Arguments
    /// * `ctx` - Application context with GPU and window access
    /// * `dt` - Delta time in seconds since last frame
    fn update(&mut self, ctx: &AppContext, dt: f32);

    /// Render a frame.
    ///
    /// Called every frame after `update()`. Record rendering commands
    /// to the provided command buffer.
    ///
    /// The framework handles:
    /// - Acquiring swapchain images
    /// - Submitting command buffers
    /// - Presenting to the screen
    ///
    /// You are responsible for:
    /// - Recording rendering commands
    /// - Blitting/copying your output to the swapchain image
    fn render(&mut self, ctx: &AppContext, frame: &mut FrameContext) -> anyhow::Result<()>;

    /// Handle window resize.
    ///
    /// Called when the window is resized. The framework automatically
    /// recreates the swapchain, but you may need to recreate other
    /// size-dependent resources.
    ///
    /// Default implementation does nothing.
    #[allow(unused_variables)]
    fn on_resize(&mut self, ctx: &mut AppContext, width: u32, height: u32) -> anyhow::Result<()> {
        Ok(())
    }

    /// Handle window events.
    ///
    /// Called for each window event. Return `true` if the event was
    /// handled and should not be processed further.
    ///
    /// Default implementation does nothing and returns `false`.
    #[allow(unused_variables)]
    fn on_event(&mut self, event: &WindowEvent) -> bool {
        false
    }

    /// Handle device events (raw input).
    ///
    /// Called for each device event. This is useful for raw mouse motion
    /// when the cursor is locked (for FPS-style controls).
    ///
    /// Default implementation does nothing.
    #[allow(unused_variables)]
    fn on_device_event(&mut self, device_id: DeviceId, event: &DeviceEvent) {}

    /// Cleanup resources before shutdown.
    ///
    /// Called when the application is about to exit. The GPU will be
    /// idle when this is called, so it's safe to destroy GPU resources.
    ///
    /// Default implementation does nothing.
    #[allow(unused_variables)]
    fn cleanup(&mut self, ctx: &mut AppContext) {}
}
