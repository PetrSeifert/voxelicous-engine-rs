//! Viewer application implementation with clipmap streaming.

use ash::vk;
use glam::Vec3;
use tracing::{error, info};
use winit::window::{CursorGrabMode, Window};

use voxelicous_app::{
    AppContext, Camera, DeviceEvent, DeviceId, FrameContext, VoxelApp, WindowEvent,
};
use voxelicous_input::{ActionMap, CursorMode, InputManager, KeyCode};
use voxelicous_render::{
    save_screenshot, CameraUniforms, ClipmapRayMarchPipeline, ClipmapRenderer, DebugMode,
    ScreenshotConfig,
};
use voxelicous_world::{ClipmapStreamingController, TerrainConfig, TerrainGenerator};

#[cfg(feature = "profiling")]
use voxelicous_profiler::QueueSizes;

/// Maximum ray marching steps per pixel.
const MAX_STEPS: u32 = 512;

/// Camera movement speed in units per second.
const CAMERA_SPEED: f32 = 30.0;

/// Camera sprint multiplier.
const CAMERA_SPRINT_MULT: f32 = 2.5;

/// Mouse sensitivity for camera rotation (radians per pixel).
const MOUSE_SENSITIVITY: f32 = 0.002;

/// Configuration for clipmap rendering (from CLI or defaults).
#[derive(Debug, Clone)]
pub struct ClipmapParams {
    pub seed: u64,
}

impl Default for ClipmapParams {
    fn default() -> Self {
        Self { seed: 42 }
    }
}

impl ClipmapParams {
    /// Parse clipmap parameters from command line arguments.
    pub fn from_args() -> Self {
        let mut params = Self::default();
        let args: Vec<String> = std::env::args().collect();

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--seed" => {
                    if i + 1 < args.len() {
                        if let Ok(v) = args[i + 1].parse() {
                            params.seed = v;
                            i += 1;
                        }
                    }
                }
                _ => {}
            }
            i += 1;
        }

        params
    }
}

/// Viewer application state with clipmap streaming.
pub struct Viewer {
    /// Clipmap streaming controller.
    clipmap: ClipmapStreamingController,
    /// GPU renderer for clipmap data.
    clipmap_renderer: ClipmapRenderer,
    /// Ray marching pipeline for clipmap rendering.
    pipeline: Option<ClipmapRayMarchPipeline>,
    /// Camera for viewing the world.
    camera: Camera,
    /// Camera yaw (rotation around Y axis) in radians.
    camera_yaw: f32,
    /// Camera pitch (rotation around X axis) in radians.
    camera_pitch: f32,
    /// Input manager for keyboard and mouse.
    input: InputManager,
    /// Screenshot configuration.
    screenshot_config: ScreenshotConfig,
    /// Clipmap parameters (stored for potential runtime adjustment).
    #[allow(dead_code)]
    clipmap_params: ClipmapParams,
    /// Whether the app should exit.
    should_exit: bool,
    /// Current debug visualization mode.
    debug_mode: DebugMode,
}

impl VoxelApp for Viewer {
    fn init(ctx: &mut AppContext) -> anyhow::Result<Self> {
        let screenshot_config = ScreenshotConfig::from_args();
        if screenshot_config.enabled {
            info!(
                "Screenshot capture enabled: {:?} frames, output pattern: {}",
                screenshot_config.frames, screenshot_config.output_pattern
            );
        }

        let clipmap_params = ClipmapParams::from_args();
        info!("Clipmap config: seed={}", clipmap_params.seed);

        // Create terrain generator for clipmap sampling
        let terrain_config = TerrainConfig {
            seed: clipmap_params.seed,
            sea_level: 64,
            terrain_scale: 50.0,
            terrain_height: 32.0,
            ..Default::default()
        };
        let generator = TerrainGenerator::new(terrain_config);

        // Create clipmap streaming controller and renderer
        let mut clipmap = ClipmapStreamingController::new(generator);
        let frames_in_flight = ctx.frames_in_flight();
        let mut clipmap_renderer = ClipmapRenderer::new(frames_in_flight);

        // Create rendering pipeline with frames_in_flight for per-frame buffers
        let pipeline = unsafe {
            let mut allocator = ctx.gpu.allocator().lock();
            ClipmapRayMarchPipeline::new(
                ctx.gpu.device(),
                &mut allocator,
                ctx.width(),
                ctx.height(),
                frames_in_flight,
            )?
        };

        info!("Clipmap ray march pipeline created");

        // Set up camera - start in the air above terrain
        let start_pos = Vec3::new(64.0, 120.0, 64.0);
        let look_at = Vec3::new(64.0, 64.0, 64.0);
        let camera = Camera::new(
            start_pos,
            look_at,
            Vec3::Y,
            60.0_f32.to_radians(),
            ctx.aspect_ratio(),
            0.1,
            1000.0,
        );

        // Calculate initial yaw and pitch from camera direction
        let dir = (look_at - start_pos).normalize();
        let camera_yaw = dir.x.atan2(dir.z);
        let camera_pitch = (-dir.y).asin();

        // Set up input manager with action bindings
        let actions = ActionMap::builder()
            .bind("move_forward", KeyCode::KeyW)
            .bind("move_forward", KeyCode::ArrowUp)
            .bind("move_back", KeyCode::KeyS)
            .bind("move_back", KeyCode::ArrowDown)
            .bind("move_left", KeyCode::KeyA)
            .bind("move_left", KeyCode::ArrowLeft)
            .bind("move_right", KeyCode::KeyD)
            .bind("move_right", KeyCode::ArrowRight)
            .bind("move_up", KeyCode::Space)
            .bind("move_down", KeyCode::ControlLeft)
            .bind("move_down", KeyCode::ControlRight)
            .bind("sprint", KeyCode::ShiftLeft)
            .bind("sprint", KeyCode::ShiftRight)
            .bind("toggle_cursor", KeyCode::Escape)
            .bind("debug_cycle", KeyCode::F3)
            .build();
        let mut input = InputManager::with_actions(actions);

        // Start with cursor locked for FPS controls
        input.set_cursor_mode(CursorMode::Locked);
        apply_cursor_mode(&ctx.window, CursorMode::Locked);

        // Initialize clipmap around the starting camera position.
        clipmap.update(start_pos);
        let dirty = clipmap.take_dirty_state();
        {
            let mut allocator = ctx.gpu.allocator().lock();
            let initial_frame_number = 0u64;
            if let Err(e) = clipmap_renderer.sync_from_controller(
                &mut allocator,
                ctx.gpu.device(),
                &clipmap,
                dirty,
                0,
                initial_frame_number,
            ) {
                error!("Failed to upload initial clipmap data: {e}");
            }
        }

        info!("Clipmap initialized");

        info!("Viewer initialized successfully!");

        Ok(Self {
            clipmap,
            clipmap_renderer,
            pipeline: Some(pipeline),
            camera,
            camera_yaw,
            camera_pitch,
            input,
            screenshot_config,
            clipmap_params,
            should_exit: false,
            debug_mode: DebugMode::default(),
        })
    }

    #[cfg_attr(feature = "profiling-tracy", tracing::instrument(level = "trace", skip_all))]
    fn update(&mut self, ctx: &AppContext, dt: f32) {
        // Update input action states (must be called before querying actions)
        self.input.update();

        // Handle cursor toggle
        if self.input.is_action_just_pressed("toggle_cursor") {
            let new_mode = match self.input.cursor_mode() {
                CursorMode::Normal => CursorMode::Locked,
                CursorMode::Locked | CursorMode::Confined => CursorMode::Normal,
            };
            self.input.set_cursor_mode(new_mode);
            apply_cursor_mode(&ctx.window, new_mode);
        }

        // Handle debug mode cycling (F3)
        if self.input.is_action_just_pressed("debug_cycle") {
            self.debug_mode = self.debug_mode.next();
            info!("Debug mode: {:?}", self.debug_mode);
        }

        // Camera rotation from mouse (only when cursor is locked)
        if self.input.cursor_mode() == CursorMode::Locked {
            let mouse_delta = self.input.mouse_raw_delta();
            self.camera_yaw -= mouse_delta.x * MOUSE_SENSITIVITY;
            self.camera_pitch += mouse_delta.y * MOUSE_SENSITIVITY;

            // Clamp pitch to prevent flipping
            self.camera_pitch = self.camera_pitch.clamp(
                -std::f32::consts::FRAC_PI_2 + 0.01,
                std::f32::consts::FRAC_PI_2 - 0.01,
            );
        }

        // Update camera direction from yaw/pitch
        let direction = Vec3::new(
            self.camera_pitch.cos() * self.camera_yaw.sin(),
            -self.camera_pitch.sin(),
            self.camera_pitch.cos() * self.camera_yaw.cos(),
        )
        .normalize();
        self.camera.direction = direction;

        // Calculate movement vectors
        let forward = Vec3::new(direction.x, 0.0, direction.z).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();

        // Calculate movement speed
        let speed = if self.input.is_action_pressed("sprint") {
            CAMERA_SPEED * CAMERA_SPRINT_MULT
        } else {
            CAMERA_SPEED
        };

        // Accumulate movement direction
        let mut movement = Vec3::ZERO;
        if self.input.is_action_pressed("move_forward") {
            movement += forward;
        }
        if self.input.is_action_pressed("move_back") {
            movement -= forward;
        }
        if self.input.is_action_pressed("move_right") {
            movement += right;
        }
        if self.input.is_action_pressed("move_left") {
            movement -= right;
        }
        if self.input.is_action_pressed("move_up") {
            movement += Vec3::Y;
        }
        if self.input.is_action_pressed("move_down") {
            movement -= Vec3::Y;
        }

        // Normalize and apply movement
        if movement != Vec3::ZERO {
            movement = movement.normalize() * speed * dt;
            self.camera.position += movement;
        }

        // End input frame (must be called at end of update)
        self.input.end_frame();

        // Update clipmap around the camera position
        self.clipmap.update(self.camera.position);

        // Report queue sizes to profiler
        #[cfg(feature = "profiling")]
        {
            let queues = QueueSizes {
                pending_page_uploads: 0,
                pending_page_unloads: 0,
                pending_page_builds: 0,
                pages_building: 0,
                resident_pages: 0,
                gpu_pages: 0,
            };
            voxelicous_profiler::report_queue_sizes(queues);
        }
    }

    #[cfg_attr(feature = "profiling-tracy", tracing::instrument(level = "trace", skip_all))]
    fn render(&mut self, ctx: &AppContext, frame: &mut FrameContext) -> anyhow::Result<()> {
        let frame_index = frame.frame_index;
        let frame_number = frame.frame_number;
        let capturing = self.screenshot_config.should_capture(frame_number);
        let camera_uniforms = self.camera.uniforms();

        self.render_sync_clipmap_buffers(ctx, frame_index, frame_number);
        self.render_record_ray_march(ctx, frame, frame_index, &camera_uniforms)?;
        self.render_record_output_blit(ctx, frame);
        self.render_record_readback(ctx, frame, capturing);
        self.render_record_present_transition(ctx, frame);

        // Handle screenshot capture
        if capturing {
            self.capture_screenshot(ctx, frame_number)?;
        }

        // Check if we should exit after capturing
        if self.screenshot_config.exit_after_capture && self.screenshot_config.all_captured(frame_number) {
            info!("All screenshots captured, requesting exit...");
            self.should_exit = true;
        }

        Ok(())
    }

    fn on_resize(&mut self, ctx: &mut AppContext, width: u32, height: u32) -> anyhow::Result<()> {
        // Recreate pipeline with new size
        unsafe {
            let mut allocator = ctx.gpu.allocator().lock();

            // Destroy old pipeline
            if let Some(old_pipeline) = self.pipeline.take() {
                old_pipeline.destroy(ctx.gpu.device(), &mut allocator)?;
            }

            // Create new pipeline with frames_in_flight
            let new_pipeline = ClipmapRayMarchPipeline::new(
                ctx.gpu.device(),
                &mut allocator,
                width,
                height,
                ctx.frames_in_flight(),
            )?;
            self.pipeline = Some(new_pipeline);
        }

        // Update camera aspect ratio
        self.camera.set_aspect(width as f32 / height as f32);

        Ok(())
    }

    fn on_event(&mut self, event: &WindowEvent) -> bool {
        if self.should_exit {
            return false;
        }
        self.input.process_window_event(event)
    }

    fn on_device_event(&mut self, _device_id: DeviceId, event: &DeviceEvent) {
        self.input.process_device_event(event);
    }

    fn cleanup(&mut self, ctx: &mut AppContext) {
        let mut allocator = ctx.gpu.allocator().lock();

        // Destroy clipmap renderer (frees all clipmap GPU resources)
        // Replace with a dummy renderer (will never be used)
        let clipmap_renderer =
            std::mem::replace(&mut self.clipmap_renderer, ClipmapRenderer::new(1));
        if let Err(e) = clipmap_renderer.destroy(&mut allocator) {
            error!("Failed to destroy clipmap renderer: {e}");
        }

        // Destroy pipeline
        if let Some(pipeline) = self.pipeline.take() {
            unsafe {
                if let Err(e) = pipeline.destroy(ctx.gpu.device(), &mut allocator) {
                    error!("Failed to destroy pipeline: {e}");
                }
            }
        }
    }
}

impl Viewer {
    #[cfg_attr(feature = "profiling-tracy", tracing::instrument(level = "trace", skip_all))]
    fn render_sync_clipmap_buffers(
        &mut self,
        ctx: &AppContext,
        frame_index: usize,
        frame_number: u64,
    ) {
        let device = ctx.gpu.device();
        let mut allocator = {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.lock_allocator").entered();
            ctx.gpu.allocator().lock()
        };

        let deferred_result = {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.process_deferred_deletions").entered();
            self.clipmap_renderer
                .process_deferred_deletions(&mut allocator, frame_number)
        };
        if let Err(e) = deferred_result {
            error!("Failed to process deferred deletions: {}", e);
        }

        let dirty = {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.take_dirty_state").entered();
            self.clipmap.take_dirty_state()
        };
        let sync_result = {
            #[cfg(feature = "profiling-tracy")]
            let _span = tracing::trace_span!("clipmap_sync.sync_from_controller").entered();
            self.clipmap_renderer.sync_from_controller(
                &mut allocator,
                device,
                &self.clipmap,
                dirty,
                frame_index,
                frame_number,
            )
        };
        if let Err(e) = sync_result {
            error!("Failed to sync clipmap GPU buffers: {e}");
        }
    }

    #[cfg_attr(feature = "profiling-tracy", tracing::instrument(level = "trace", skip_all))]
    fn render_record_ray_march(
        &self,
        ctx: &AppContext,
        frame: &FrameContext,
        frame_index: usize,
        camera_uniforms: &CameraUniforms,
    ) -> anyhow::Result<()> {
        let device = ctx.gpu.device();
        let cmd = frame.command_buffer;
        let pipeline = self.pipeline.as_ref().expect("Pipeline should exist");

        unsafe {
            pipeline.record(
                device,
                cmd,
                camera_uniforms,
                &self.clipmap_renderer,
                MAX_STEPS,
                frame_index,
                self.debug_mode,
            )?;
        }

        Ok(())
    }

    #[cfg_attr(feature = "profiling-tracy", tracing::instrument(level = "trace", skip_all))]
    fn render_record_output_blit(&self, ctx: &AppContext, frame: &FrameContext) {
        let device = ctx.gpu.device();
        let cmd = frame.command_buffer;
        let pipeline = self.pipeline.as_ref().expect("Pipeline should exist");

        unsafe {
            // Transition output image for transfer
            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(pipeline.output_image().image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let swapchain_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(frame.swapchain_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let barriers = [barrier, swapchain_barrier];
            let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            device.cmd_pipeline_barrier2(cmd, &dependency_info);

            // Blit from pipeline output to swapchain
            let (out_w, out_h) = pipeline.dimensions();
            let blit = vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: out_w as i32,
                        y: out_h as i32,
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: ctx.width() as i32,
                        y: ctx.height() as i32,
                        z: 1,
                    },
                ],
            };

            device.cmd_blit_image(
                cmd,
                pipeline.output_image().image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                frame.swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::LINEAR,
            );
        }
    }

    #[cfg_attr(feature = "profiling-tracy", tracing::instrument(level = "trace", skip_all))]
    fn render_record_readback(
        &self,
        ctx: &AppContext,
        frame: &FrameContext,
        capturing: bool,
    ) {
        if !capturing {
            return;
        }

        let device = ctx.gpu.device();
        let cmd = frame.command_buffer;
        let pipeline = self.pipeline.as_ref().expect("Pipeline should exist");

        unsafe {
            pipeline.record_readback_from_transfer_src(device, cmd);
        }
    }

    #[cfg_attr(feature = "profiling-tracy", tracing::instrument(level = "trace", skip_all))]
    fn render_record_present_transition(&self, ctx: &AppContext, frame: &FrameContext) {
        let device = ctx.gpu.device();
        let cmd = frame.command_buffer;

        unsafe {
            // Transition swapchain image for present
            let present_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(frame.swapchain_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let dependency_info = vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&present_barrier));
            device.cmd_pipeline_barrier2(cmd, &dependency_info);
        }
    }

    fn capture_screenshot(&self, ctx: &AppContext, frame_number: u64) -> anyhow::Result<()> {
        ctx.gpu.wait_idle()?;

        let pipeline = self.pipeline.as_ref().expect("Pipeline should exist");
        let (width, height) = pipeline.dimensions();

        match pipeline.read_output() {
            Ok(data) => {
                let output_path = self.screenshot_config.output_path(frame_number);
                if let Err(e) = save_screenshot(data, width, height, &output_path) {
                    error!("Failed to save screenshot: {e}");
                }
            }
            Err(e) => {
                error!("Failed to read output for screenshot: {e}");
            }
        }

        Ok(())
    }
}

/// Apply cursor mode to the window.
fn apply_cursor_mode(window: &Window, mode: CursorMode) {
    let (grab_mode, visible) = match mode {
        CursorMode::Normal => (CursorGrabMode::None, true),
        CursorMode::Confined => (CursorGrabMode::Confined, true),
        CursorMode::Locked => (CursorGrabMode::Locked, false),
    };

    // Try to set cursor grab mode, falling back if not supported
    if let Err(e) = window.set_cursor_grab(grab_mode) {
        // Fall back to confined mode if locked is not supported
        if grab_mode == CursorGrabMode::Locked {
            if let Err(e2) = window.set_cursor_grab(CursorGrabMode::Confined) {
                tracing::warn!("Failed to confine cursor: {e2}");
            }
        } else {
            tracing::warn!("Failed to set cursor grab mode: {e}");
        }
    }

    window.set_cursor_visible(visible);
}
