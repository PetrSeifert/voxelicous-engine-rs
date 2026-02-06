//! Application runner and event loop.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ash::vk;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;
#[cfg(feature = "profiling-tracy")]
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use voxelicous_gpu::command::submit_command_buffers;
use voxelicous_gpu::sync::{reset_fence, wait_for_fence};
use voxelicous_gpu::GpuContextBuilder;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::app::VoxelApp;
use crate::context::AppContext;
use crate::frame::FrameContext;

#[cfg(feature = "profiling")]
use voxelicous_profiler::{profile_scope, EventCategory};

/// Application configuration.
#[derive(Clone)]
pub struct AppConfig {
    /// Window title.
    pub title: String,
    /// Initial window width.
    pub width: u32,
    /// Initial window height.
    pub height: u32,
    /// Target frames per second (None for unlimited).
    pub target_fps: Option<u32>,
    /// Enable vsync.
    pub vsync: bool,
    /// Enable Vulkan validation layers (default: debug builds only).
    pub validation: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            title: "Voxelicous Engine".to_string(),
            width: 1280,
            height: 720,
            target_fps: None,
            vsync: false,
            validation: cfg!(debug_assertions),
        }
    }
}

impl AppConfig {
    /// Create a new config with the given title.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Default::default()
        }
    }

    /// Set the window dimensions.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set the target FPS.
    pub fn with_target_fps(mut self, fps: u32) -> Self {
        self.target_fps = Some(fps);
        self
    }

    /// Enable or disable vsync.
    pub fn with_vsync(mut self, vsync: bool) -> Self {
        self.vsync = vsync;
        self
    }

    /// Enable or disable validation layers.
    pub fn with_validation(mut self, validation: bool) -> Self {
        self.validation = validation;
        self
    }
}

/// Run a VoxelApp with the given configuration.
///
/// This function initializes logging, creates the window and GPU context,
/// and runs the event loop until the application exits.
pub fn run_app<A: VoxelApp + 'static>(config: AppConfig) -> anyhow::Result<()> {
    // Initialize logging
    #[cfg(feature = "profiling-tracy")]
    {
        let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new(
                "info,voxelicous_app=trace,voxelicous_world=trace,voxelicous_render=trace,voxelicous_gpu=trace,voxelicous_viewer=trace,voxelicous_editor=trace,voxelicous_benchmark=trace",
            )
        });
        let tracy_layer = tracing_tracy::TracyLayer::default();

        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer())
            .with(tracy_layer)
            .init();
    }
    #[cfg(not(feature = "profiling-tracy"))]
    {
        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .init();
    }

    // Initialize profiler
    #[cfg(feature = "profiling")]
    {
        voxelicous_profiler::init();
        info!(
            "Profiler initialized on port {}",
            voxelicous_profiler::DEFAULT_PORT
        );
    }

    info!("{} starting...", config.title);

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut runner = AppRunner::<A> {
        config,
        state: None,
    };

    if let Err(e) = event_loop.run_app(&mut runner) {
        error!("Event loop error: {e}");
    }

    Ok(())
}

/// Internal application runner that implements winit's ApplicationHandler.
struct AppRunner<A: VoxelApp> {
    config: AppConfig,
    state: Option<AppState<A>>,
}

/// Internal application state.
struct AppState<A: VoxelApp> {
    ctx: AppContext,
    app: A,
    target_frame_time: Option<Duration>,
    // FPS tracking
    min_fps: f64,
    max_fps: f64,
    fps_sum: f64,
}

impl<A: VoxelApp + 'static> ApplicationHandler for AppRunner<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        info!("Creating application state...");

        match self.create_state(event_loop) {
            Ok(state) => {
                self.state = Some(state);
                info!("Application ready!");
            }
            Err(e) => {
                error!("Failed to initialize application: {e}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Let the app handle the event first
        if let Some(state) = &mut self.state {
            if state.app.on_event(&event) {
                return;
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                info!("Close requested");
                if let Some(mut state) = self.state.take() {
                    state.cleanup();
                }
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    if let Err(e) = state.render_frame() {
                        error!("Render error: {e}");
                    }
                    state.ctx.window.request_redraw();
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    if let Err(e) = state.handle_resize(size.width, size.height) {
                        error!("Resize error: {e}");
                    }
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(state) = &mut self.state {
            state.app.on_device_event(device_id, &event);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.ctx.window.request_redraw();
        }
    }
}

impl<A: VoxelApp + 'static> AppRunner<A> {
    fn create_state(&self, event_loop: &ActiveEventLoop) -> anyhow::Result<AppState<A>> {
        // Create window
        let window_attrs = Window::default_attributes()
            .with_title(&self.config.title)
            .with_inner_size(PhysicalSize::new(self.config.width, self.config.height));

        let window = Arc::new(event_loop.create_window(window_attrs)?);

        // Create GPU context
        let gpu = GpuContextBuilder::new()
            .app_name(&self.config.title)
            .validation(self.config.validation)
            .build()?;

        info!("GPU: {}", gpu.capabilities().summary());

        // Create app context
        let mut ctx = unsafe { AppContext::new(window, gpu, self.config.vsync)? };

        // Initialize the application
        let app = A::init(&mut ctx)?;

        let target_frame_time = self
            .config
            .target_fps
            .map(|fps| Duration::from_nanos(1_000_000_000 / fps as u64));

        Ok(AppState {
            ctx,
            app,
            target_frame_time,
            min_fps: f64::MAX,
            max_fps: 0.0,
            fps_sum: 0.0,
        })
    }
}

impl<A: VoxelApp> AppState<A> {
    #[cfg_attr(feature = "profiling-tracy", tracing::instrument(level = "trace", skip_all))]
    fn render_frame(&mut self) -> anyhow::Result<()> {
        #[cfg(feature = "profiling")]
        profile_scope!(EventCategory::Frame);

        #[cfg(feature = "profiling-tracy")]
        let _frame_span = tracing::trace_span!(
            "frame.sections",
            frame_index = self.ctx.current_frame_index as u32,
            frame_number = self.ctx.frame_count
        )
        .entered();

        let frame_start = Instant::now();

        // Calculate delta time
        #[allow(unused_variables)]
        let (dt, fps) = {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("frame.timing").entered();

            let now = Instant::now();
            let dt = now.duration_since(self.ctx.last_frame_time).as_secs_f32();
            self.ctx.last_frame_time = now;

            // Update FPS tracking
            let fps = if dt > 0.0 {
                let fps = 1.0 / dt as f64;
                self.min_fps = self.min_fps.min(fps);
                self.max_fps = self.max_fps.max(fps);
                self.fps_sum += fps;
                fps as f32
            } else {
                0.0
            };

            (dt, fps)
        };

        // Update the application
        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("frame.update").entered();

            #[cfg(feature = "profiling")]
            profile_scope!(EventCategory::FrameUpdate);
            self.app.update(&self.ctx, dt);
        }

        let device = self.ctx.gpu.device();
        let frame_data = &self.ctx.frames[self.ctx.current_frame_index];

        // GPU synchronization: wait for previous frame and acquire next image
        let image_index = {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("frame.gpu_sync").entered();

            #[cfg(feature = "profiling")]
            profile_scope!(EventCategory::GpuSync);

            unsafe {
                // Wait for this frame slot's fence
                {
                    #[cfg(feature = "profiling-tracy")]
                    let _span = tracing::trace_span!("frame.gpu_sync.wait_fence").entered();
                    wait_for_fence(device, frame_data.in_flight_fence, u64::MAX)?;
                }

                // Acquire swapchain image
                let (image_index, _suboptimal) = {
                    #[cfg(feature = "profiling-tracy")]
                    let _span = tracing::trace_span!("frame.gpu_sync.acquire_image").entered();
                    self.ctx.swapchain.acquire_next_image(
                        &self.ctx.surface.swapchain_loader,
                        frame_data.image_available,
                        u64::MAX,
                    )?
                };

                // Reset fence after successful acquire
                {
                    #[cfg(feature = "profiling-tracy")]
                    let _span = tracing::trace_span!("frame.gpu_sync.reset_fence").entered();
                    reset_fence(device, frame_data.in_flight_fence)?;
                }

                image_index
            }
        };

        // Render: record command buffer
        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("frame.record").entered();

            #[cfg(feature = "profiling")]
            profile_scope!(EventCategory::FrameRender);

            unsafe {
                // Reset and begin command buffer
                {
                    #[cfg(feature = "profiling-tracy")]
                    let _span = tracing::trace_span!("frame.record.begin_cmd").entered();
                    device.reset_command_buffer(
                        frame_data.command_buffer,
                        vk::CommandBufferResetFlags::empty(),
                    )?;

                    let begin_info = vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                    device.begin_command_buffer(frame_data.command_buffer, &begin_info)?;
                }

                // Create frame context
                let mut frame_ctx = {
                    #[cfg(feature = "profiling-tracy")]
                    let _span = tracing::trace_span!("frame.record.build_context").entered();
                    FrameContext::new(
                        frame_data.command_buffer,
                        image_index,
                        self.ctx.swapchain.images[image_index as usize],
                        dt,
                        self.ctx.frame_count,
                        self.ctx.current_frame_index,
                    )
                };

                // Render the frame
                {
                    #[cfg(feature = "profiling-tracy")]
                    let _span = tracing::trace_span!("frame.record.app_render").entered();
                    self.app.render(&self.ctx, &mut frame_ctx)?;
                }

                // End command buffer
                {
                    #[cfg(feature = "profiling-tracy")]
                    let _span = tracing::trace_span!("frame.record.end_cmd").entered();
                    device.end_command_buffer(frame_data.command_buffer)?;
                }
            }
        }

        // Get the render finished semaphore for this swapchain image
        let render_finished = self.ctx.render_finished_semaphores[image_index as usize];

        // Submit command buffer to GPU
        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("frame.submit").entered();

            #[cfg(feature = "profiling")]
            profile_scope!(EventCategory::GpuSubmit);

            let wait_semaphores = [frame_data.image_available];
            let wait_stages = [vk::PipelineStageFlags::TRANSFER];
            let signal_semaphores = [render_finished];
            let command_buffers = [frame_data.command_buffer];

            unsafe {
                submit_command_buffers(
                    device,
                    self.ctx.gpu.graphics_queue(),
                    &command_buffers,
                    &wait_semaphores,
                    &wait_stages,
                    &signal_semaphores,
                    frame_data.in_flight_fence,
                )?;
            }
        }

        // Present to swapchain
        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("frame.present").entered();

            #[cfg(feature = "profiling")]
            profile_scope!(EventCategory::FramePresent);

            unsafe {
                self.ctx.swapchain.present(
                    &self.ctx.surface.swapchain_loader,
                    self.ctx.gpu.graphics_queue(),
                    image_index,
                    &[render_finished],
                )?;
            }
        }

        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("frame.advance").entered();
            self.ctx.current_frame_index = (self.ctx.current_frame_index + 1) % self.ctx.frames.len();
            self.ctx.frame_count += 1;
        }

        // Frame pacing
        if let Some(target) = self.target_frame_time {
            let elapsed = frame_start.elapsed();
            if elapsed < target {
                #[cfg(feature = "profiling-tracy")]
                let _span = tracing::trace_span!("frame.pacing").entered();
                thread::sleep(target - elapsed);
            }
        }

        // Report frame to profiler
        {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("frame.profiler.end_frame").entered();
            #[cfg(feature = "profiling")]
            voxelicous_profiler::end_frame(self.ctx.frame_count, fps, dt * 1000.0);
        }

        Ok(())
    }

    fn handle_resize(&mut self, width: u32, height: u32) -> anyhow::Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }

        unsafe {
            self.ctx.gpu.wait_idle()?;
            self.ctx.recreate_swapchain(width, height)?;
        }

        // Notify the application
        self.app.on_resize(&mut self.ctx, width, height)?;

        info!("Resized to {}x{}", width, height);
        Ok(())
    }

    fn cleanup(&mut self) {
        // Shutdown profiler
        #[cfg(feature = "profiling")]
        voxelicous_profiler::shutdown();

        // Print FPS statistics
        if self.ctx.frame_count > 0 {
            let avg_fps = self.fps_sum / self.ctx.frame_count as f64;
            info!("FPS Statistics:");
            info!("  Min: {:.1}", self.min_fps);
            info!("  Max: {:.1}", self.max_fps);
            info!("  Avg: {:.1}", avg_fps);
            info!("  Total frames: {}", self.ctx.frame_count);
        }

        info!("Starting cleanup...");
        unsafe {
            if let Err(e) = self.ctx.gpu.wait_idle() {
                error!("Failed to wait idle: {e}");
            }

            // Let the app cleanup first
            self.app.cleanup(&mut self.ctx);

            // Then cleanup context resources
            self.ctx.cleanup();

            info!("Cleanup complete");
        }
    }
}
