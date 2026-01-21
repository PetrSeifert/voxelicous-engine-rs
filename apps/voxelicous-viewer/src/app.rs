//! Viewer application implementation with chunk streaming.

use ash::vk;
use glam::Vec3;
use tracing::{debug, error, info};
use winit::window::{CursorGrabMode, Window};

use voxelicous_app::{AppContext, Camera, DeviceEvent, DeviceId, FrameContext, VoxelApp, WindowEvent};
use voxelicous_core::coords::ChunkPos;
use voxelicous_input::{ActionMap, CursorMode, InputManager, KeyCode};
use voxelicous_render::{save_screenshot, ScreenshotConfig, WorldRayMarchPipeline, WorldRenderer};
use voxelicous_world::{ChunkState, StreamingConfig, TerrainConfig, World};

/// Maximum ray marching steps per chunk.
const MAX_STEPS: u32 = 128;

/// Default maximum chunks to keep on GPU.
const DEFAULT_MAX_GPU_CHUNKS: usize = 64;

/// Camera movement speed in units per second.
const CAMERA_SPEED: f32 = 30.0;

/// Camera sprint multiplier.
const CAMERA_SPRINT_MULT: f32 = 2.5;

/// Mouse sensitivity for camera rotation (radians per pixel).
const MOUSE_SENSITIVITY: f32 = 0.002;

/// Configuration for chunk streaming (from CLI or defaults).
#[derive(Debug, Clone)]
pub struct StreamingParams {
    pub load_radius: i32,
    pub unload_radius: i32,
    pub vertical_radius: i32,
    pub max_gen_per_frame: usize,
    pub seed: u64,
}

impl Default for StreamingParams {
    fn default() -> Self {
        Self {
            load_radius: 3,
            unload_radius: 5,
            vertical_radius: 2,
            max_gen_per_frame: 2,
            seed: 42,
        }
    }
}

impl StreamingParams {
    /// Parse streaming parameters from command line arguments.
    pub fn from_args() -> Self {
        let mut params = Self::default();
        let args: Vec<String> = std::env::args().collect();

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--load-radius" => {
                    if i + 1 < args.len() {
                        if let Ok(v) = args[i + 1].parse() {
                            params.load_radius = v;
                            i += 1;
                        }
                    }
                }
                "--unload-radius" => {
                    if i + 1 < args.len() {
                        if let Ok(v) = args[i + 1].parse() {
                            params.unload_radius = v;
                            i += 1;
                        }
                    }
                }
                "--vertical-radius" => {
                    if i + 1 < args.len() {
                        if let Ok(v) = args[i + 1].parse() {
                            params.vertical_radius = v;
                            i += 1;
                        }
                    }
                }
                "--max-gen-per-frame" => {
                    if i + 1 < args.len() {
                        if let Ok(v) = args[i + 1].parse() {
                            params.max_gen_per_frame = v;
                            i += 1;
                        }
                    }
                }
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

        // Ensure unload radius is greater than load radius
        if params.unload_radius <= params.load_radius {
            params.unload_radius = params.load_radius + 4;
        }

        params
    }
}

/// Viewer application state with chunk streaming.
pub struct Viewer {
    /// World containing chunk manager and streamer.
    world: World,
    /// GPU renderer for world chunks.
    world_renderer: WorldRenderer,
    /// Ray marching pipeline for world rendering.
    pipeline: Option<WorldRayMarchPipeline>,
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
    /// Streaming parameters (stored for potential runtime adjustment).
    #[allow(dead_code)]
    streaming_params: StreamingParams,
    /// Whether the app should exit.
    should_exit: bool,
    /// Chunks waiting to be uploaded to GPU.
    pending_uploads: Vec<ChunkPos>,
    /// Chunks to be removed from GPU.
    pending_unloads: Vec<ChunkPos>,
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

        let streaming_params = StreamingParams::from_args();
        info!(
            "Streaming config: load_radius={}, unload_radius={}, vertical_radius={}, max_gen={}",
            streaming_params.load_radius,
            streaming_params.unload_radius,
            streaming_params.vertical_radius,
            streaming_params.max_gen_per_frame
        );

        // Create terrain and streaming configuration
        let terrain_config = TerrainConfig {
            seed: streaming_params.seed,
            sea_level: 64,
            terrain_scale: 50.0,
            terrain_height: 32.0,
            ..Default::default()
        };

        let streaming_config = StreamingConfig {
            load_radius: streaming_params.load_radius,
            unload_radius: streaming_params.unload_radius,
            vertical_radius: streaming_params.vertical_radius,
            max_gen_per_update: streaming_params.max_gen_per_frame,
            max_compress_per_update: streaming_params.max_gen_per_frame * 2,
        };

        // Create world with streaming
        let mut world = World::with_config(terrain_config, streaming_config, 1024);

        // Create world renderer
        let mut world_renderer = WorldRenderer::new(DEFAULT_MAX_GPU_CHUNKS);

        // Create rendering pipeline
        let pipeline = unsafe {
            let mut allocator = ctx.gpu.allocator().lock();
            WorldRayMarchPipeline::new(ctx.gpu.device(), &mut allocator, ctx.width(), ctx.height())?
        };

        info!("World ray march pipeline created");

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
            .build();
        let mut input = InputManager::with_actions(actions);

        // Start with cursor locked for FPS controls
        input.set_cursor_mode(CursorMode::Locked);
        apply_cursor_mode(&ctx.window, CursorMode::Locked);

        // Generate initial chunks around camera position
        let camera_chunk = ChunkPos::new(
            (start_pos.x / 32.0).floor() as i32,
            (start_pos.y / 32.0).floor() as i32,
            (start_pos.z / 32.0).floor() as i32,
        );

        // Start with a small number of chunks to avoid GPU overload
        // The streaming system will load more as needed
        info!("Generating initial chunks around {:?}...", camera_chunk);
        world.generate_initial_chunks(camera_chunk, 1, 1); // 3x3x3 = 27 chunks
        info!("Generated {} initial chunks", world.chunk_count());

        // Collect initial chunks for GPU upload
        let pending_uploads: Vec<ChunkPos> = world.chunks_ready_for_upload();
        info!("{} chunks ready for GPU upload", pending_uploads.len());

        // Upload initial chunks to GPU
        {
            let mut allocator = ctx.gpu.allocator().lock();
            for pos in &pending_uploads {
                if let Some(dag) = get_chunk_dag(&world, *pos) {
                    if let Err(e) =
                        world_renderer.upload_chunk(&mut allocator, ctx.gpu.device(), *pos, &dag)
                    {
                        error!("Failed to upload chunk {:?}: {}", pos, e);
                    }
                }
            }

            // Rebuild chunk info buffer
            if let Err(e) =
                world_renderer.rebuild_chunk_info_buffer(&mut allocator, ctx.gpu.device())
            {
                error!("Failed to rebuild chunk info buffer: {}", e);
            }
        }

        info!(
            "Uploaded {} chunks to GPU ({:.2} MB)",
            world_renderer.chunk_count(),
            world_renderer.gpu_memory_usage() as f64 / (1024.0 * 1024.0)
        );

        info!("Viewer initialized successfully!");

        Ok(Self {
            world,
            world_renderer,
            pipeline: Some(pipeline),
            camera,
            camera_yaw,
            camera_pitch,
            input,
            screenshot_config,
            streaming_params,
            should_exit: false,
            pending_uploads: Vec::new(),
            pending_unloads: Vec::new(),
        })
    }

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

        // Update world streaming based on camera position
        let (needs_upload, unloaded) = self.world.update(self.camera.position);

        // Store pending operations for render phase
        self.pending_uploads.extend(needs_upload);
        self.pending_unloads.extend(unloaded);

        // Log streaming activity periodically
        if ctx.frame_count % 60 == 0
            && (!self.pending_uploads.is_empty() || !self.pending_unloads.is_empty())
        {
            debug!(
                "Streaming: {} uploads pending, {} unloads, {} total chunks",
                self.pending_uploads.len(),
                self.pending_unloads.len(),
                self.world.chunk_count()
            );
        }
    }

    fn render(&mut self, ctx: &AppContext, frame: &mut FrameContext) -> anyhow::Result<()> {
        let device = ctx.gpu.device();
        let cmd = frame.command_buffer;
        let camera_uniforms = self.camera.uniforms();

        // Process pending GPU operations before rendering
        // Limit uploads/unloads per frame to prevent GPU stalls
        const MAX_UPLOADS_PER_FRAME: usize = 4;
        const MAX_UNLOADS_PER_FRAME: usize = 8;

        {
            let mut allocator = ctx.gpu.allocator().lock();

            // Upload new chunks (limited per frame)
            let upload_count = self.pending_uploads.len().min(MAX_UPLOADS_PER_FRAME);
            let uploads: Vec<_> = self.pending_uploads.drain(..upload_count).collect();

            for pos in uploads {
                if !self.world_renderer.has_chunk(pos) {
                    if let Some(dag) = get_chunk_dag(&self.world, pos) {
                        if let Err(e) =
                            self.world_renderer
                                .upload_chunk(&mut allocator, device, pos, &dag)
                        {
                            error!("Failed to upload chunk {:?}: {}", pos, e);
                        }
                    }
                }
            }

            // Remove unloaded chunks from GPU (limited per frame)
            let unload_count = self.pending_unloads.len().min(MAX_UNLOADS_PER_FRAME);
            let unloads: Vec<_> = self.pending_unloads.drain(..unload_count).collect();

            for pos in unloads {
                if let Err(e) = self
                    .world_renderer
                    .remove_chunk(&mut allocator, device, pos)
                {
                    error!("Failed to remove chunk {:?}: {}", pos, e);
                }
            }

            // Rebuild chunk info buffer if dirty
            if let Err(e) = self
                .world_renderer
                .rebuild_chunk_info_buffer(&mut allocator, device)
            {
                error!("Failed to rebuild chunk info buffer: {}", e);
            }
        }

        let pipeline = self.pipeline.as_ref().expect("Pipeline should exist");

        unsafe {
            // Record rendering commands
            pipeline.record(
                device,
                cmd,
                &camera_uniforms,
                &self.world_renderer,
                MAX_STEPS,
            )?;

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

            // Record readback if capturing this frame
            let capturing = self.screenshot_config.should_capture(frame.frame_number);
            if capturing {
                pipeline.record_readback_from_transfer_src(device, cmd);
            }

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

        // Handle screenshot capture
        if self.screenshot_config.should_capture(frame.frame_number) {
            self.capture_screenshot(ctx, frame.frame_number)?;
        }

        // Check if we should exit after capturing
        if self.screenshot_config.exit_after_capture
            && self.screenshot_config.all_captured(frame.frame_number)
        {
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

            // Create new pipeline
            let new_pipeline =
                WorldRayMarchPipeline::new(ctx.gpu.device(), &mut allocator, width, height)?;
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

        // Destroy world renderer (frees all chunk GPU resources)
        let world_renderer = std::mem::replace(&mut self.world_renderer, WorldRenderer::new(0));
        if let Err(e) = world_renderer.destroy(&mut allocator) {
            error!("Failed to destroy world renderer: {e}");
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

/// Helper to get the compressed DAG from a chunk.
fn get_chunk_dag(world: &World, pos: ChunkPos) -> Option<voxelicous_voxel::SvoDag> {
    let mut result = None;
    world.chunk_manager.with_chunk(pos, |chunk| {
        if chunk.state == ChunkState::Compressed {
            result = chunk.dag.clone();
        }
    });
    result
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
