//! Viewer application implementation.

use ash::vk;
use glam::Vec3;
use tracing::{error, info};

use voxelicous_app::{AppContext, Camera, FrameContext, UnifiedPipeline, VoxelApp, WindowEvent};
use voxelicous_render::{save_screenshot, GpuSvoDag, ScreenshotConfig};

use crate::terrain::{generate_world, WorldParams};

/// Maximum ray marching steps.
const MAX_STEPS: u32 = 256;

/// Viewer application state.
pub struct Viewer {
    gpu_dag: Option<GpuSvoDag>,
    pipeline: Option<UnifiedPipeline>,
    camera: Camera,
    screenshot_config: ScreenshotConfig,
    world_params: WorldParams,
    should_exit: bool,
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

        let world_params = WorldParams::default();

        // Create rendering pipeline
        let (pipeline, render_path) = unsafe {
            let mut allocator = ctx.gpu.allocator().lock();
            UnifiedPipeline::new(
                &ctx.gpu,
                &mut allocator,
                ctx.width(),
                ctx.height(),
                world_params.world_depth,
            )?
        };

        info!("Rendering pipeline created: {:?}", render_path);

        // Build acceleration structures for hardware RT
        if pipeline.is_hardware_rt() {
            build_acceleration_structures(ctx, &pipeline)?;
        }

        // Generate terrain
        let world = generate_world(&world_params);

        // Upload DAG to GPU
        let gpu_dag = {
            let mut allocator = ctx.gpu.allocator().lock();
            GpuSvoDag::upload(&mut allocator, ctx.gpu.device(), &world.dag)?
        };

        info!("DAG uploaded to GPU");

        // Set up camera
        let world_center = Vec3::new(64.0, 32.0, 64.0);
        let camera = Camera::new(
            Vec3::new(200.0, 80.0, 200.0),
            world_center,
            Vec3::Y,
            60.0_f32.to_radians(),
            ctx.aspect_ratio(),
            0.1,
            1000.0,
        );

        info!("Viewer initialized successfully!");

        Ok(Self {
            gpu_dag: Some(gpu_dag),
            pipeline: Some(pipeline),
            camera,
            screenshot_config,
            world_params,
            should_exit: false,
        })
    }

    fn update(&mut self, ctx: &AppContext, _dt: f32) {
        // Animate camera - orbit around the world center
        let time = ctx.frame_count as f32 * 0.005;
        let radius = 180.0;
        let world_center = Vec3::new(64.0, 32.0, 64.0);
        let camera_pos = Vec3::new(
            world_center.x + radius * time.cos(),
            80.0,
            world_center.z + radius * time.sin(),
        );
        self.camera.set_position(camera_pos);
        self.camera.look_at(world_center);
    }

    fn render(&mut self, ctx: &AppContext, frame: &mut FrameContext) -> anyhow::Result<()> {
        let device = ctx.gpu.device();
        let cmd = frame.command_buffer;
        let camera_uniforms = self.camera.uniforms();

        let pipeline = self.pipeline.as_ref().expect("Pipeline should exist");
        let gpu_dag = self.gpu_dag.as_ref().expect("GPU DAG should exist");

        unsafe {
            // Record rendering commands
            pipeline.record(&ctx.gpu, cmd, &camera_uniforms, gpu_dag, MAX_STEPS)?;

            // Transition output image for transfer
            let src_stage = pipeline.output_stage();

            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src_stage)
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
                old_pipeline.destroy(&ctx.gpu, &mut allocator)?;
            }

            // Create new pipeline
            let (new_pipeline, _render_path) = UnifiedPipeline::new(
                &ctx.gpu,
                &mut allocator,
                width,
                height,
                self.world_params.world_depth,
            )?;
            self.pipeline = Some(new_pipeline);
        }

        // Rebuild acceleration structures for hardware RT
        let pipeline = self.pipeline.as_ref().expect("Pipeline should exist");
        if pipeline.is_hardware_rt() {
            build_acceleration_structures(ctx, pipeline)?;
        }

        // Update camera aspect ratio
        self.camera.set_aspect(width as f32 / height as f32);

        Ok(())
    }

    fn on_event(&mut self, _event: &WindowEvent) -> bool {
        if self.should_exit {
            return false;
        }
        false
    }

    fn cleanup(&mut self, ctx: &mut AppContext) {
        let mut allocator = ctx.gpu.allocator().lock();

        if let Some(gpu_dag) = self.gpu_dag.take() {
            if let Err(e) = gpu_dag.destroy(&mut allocator) {
                error!("Failed to destroy GPU DAG: {e}");
            }
        }

        if let Some(pipeline) = self.pipeline.take() {
            unsafe {
                if let Err(e) = pipeline.destroy(&ctx.gpu, &mut allocator) {
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

/// Build acceleration structures for hardware ray tracing.
fn build_acceleration_structures(
    ctx: &AppContext,
    pipeline: &UnifiedPipeline,
) -> anyhow::Result<()> {
    info!("Building acceleration structures...");

    unsafe {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let as_cmd = ctx.gpu.device().allocate_command_buffers(&alloc_info)?[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        ctx.gpu.device().begin_command_buffer(as_cmd, &begin_info)?;

        pipeline.build_acceleration_structures(&ctx.gpu, as_cmd);

        ctx.gpu.device().end_command_buffer(as_cmd)?;

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&as_cmd));
        ctx.gpu.device().queue_submit(
            ctx.gpu.graphics_queue(),
            &[submit_info],
            vk::Fence::null(),
        )?;
        ctx.gpu.wait_idle()?;

        ctx.gpu
            .device()
            .free_command_buffers(ctx.command_pool, &[as_cmd]);
    }

    info!("Acceleration structures built");
    Ok(())
}
