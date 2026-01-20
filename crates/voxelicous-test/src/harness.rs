//! Test harness for headless rendering and visual regression testing.
//!
//! This module provides infrastructure for testing the rendering pipeline
//! without a display, using visual regression tests to verify correctness.

use ash::vk;
use glam::Vec3;
use image::{ImageBuffer, Rgba};
use std::path::Path;

use voxelicous_gpu::{GpuContext, GpuContextBuilder};
use voxelicous_render::{Camera, CameraUniforms, GpuSvoDag, RayMarchPipeline};
use voxelicous_voxel::{SparseVoxelOctree, SvoDag, VoxelStorage};

use crate::{Result, TestError, VisualTestConfig};

/// Headless renderer for testing.
///
/// Provides a GPU context and rendering pipeline for off-screen rendering.
pub struct HeadlessRenderer {
    context: GpuContext,
    pipeline: RayMarchPipeline,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    width: u32,
    height: u32,
}

impl HeadlessRenderer {
    /// Create a new headless renderer.
    ///
    /// # Arguments
    /// * `width` - Output image width
    /// * `height` - Output image height
    pub fn new(width: u32, height: u32) -> Result<Self> {
        // Create headless GPU context
        let context = GpuContextBuilder::new()
            .app_name("voxelicous-test")
            .validation(true)
            .build()
            .map_err(|e| TestError::Gpu(e.to_string()))?;

        let device = context.device();
        let graphics_queue_family = context.graphics_queue_family();

        // Create command pool
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .map_err(|e| TestError::Gpu(format!("Failed to create command pool: {e}")))?
        };

        // Allocate command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| TestError::Gpu(format!("Failed to allocate command buffer: {e}")))?
        };
        let command_buffer = command_buffers[0];

        // Create fence
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe {
            device
                .create_fence(&fence_info, None)
                .map_err(|e| TestError::Gpu(format!("Failed to create fence: {e}")))?
        };

        // Create ray march pipeline
        let mut allocator = context.allocator().lock();
        let pipeline = unsafe {
            RayMarchPipeline::new(device, &mut allocator, width, height)
                .map_err(|e| TestError::Gpu(e.to_string()))?
        };
        drop(allocator);

        Ok(Self {
            context,
            pipeline,
            command_pool,
            command_buffer,
            fence,
            width,
            height,
        })
    }

    /// Render an SVO and return the resulting image.
    ///
    /// # Arguments
    /// * `svo` - The sparse voxel octree to render
    /// * `camera` - Camera configuration
    pub fn render_svo(
        &mut self,
        svo: &SparseVoxelOctree,
        camera: &Camera,
    ) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
        let dag = SvoDag::from_svo(svo);
        self.render_dag(&dag, camera)
    }

    /// Render a DAG and return the resulting image.
    ///
    /// # Arguments
    /// * `dag` - The SVO-DAG to render
    /// * `camera` - Camera configuration
    pub fn render_dag(
        &mut self,
        dag: &SvoDag,
        camera: &Camera,
    ) -> Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
        let device = self.context.device();
        let mut allocator = self.context.allocator().lock();

        // Upload DAG to GPU
        let gpu_dag = GpuSvoDag::upload(&mut allocator, device, dag)
            .map_err(|e| TestError::Gpu(e.to_string()))?;

        // Create camera uniforms
        let camera_uniforms = CameraUniforms::from(camera);

        // Begin command buffer
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .map_err(|e| TestError::Gpu(format!("Failed to begin command buffer: {e}")))?;

            // Record ray march commands
            self.pipeline
                .record(device, self.command_buffer, &camera_uniforms, &gpu_dag, 256)
                .map_err(|e| TestError::Gpu(e.to_string()))?;

            // Record readback commands
            self.pipeline.record_readback(device, self.command_buffer);

            // End command buffer
            device
                .end_command_buffer(self.command_buffer)
                .map_err(|e| TestError::Gpu(format!("Failed to end command buffer: {e}")))?;

            // Submit
            let submit_info = vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&self.command_buffer));

            device
                .queue_submit(self.context.graphics_queue(), &[submit_info], self.fence)
                .map_err(|e| TestError::Gpu(format!("Failed to submit queue: {e}")))?;

            // Wait for completion
            device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| TestError::Gpu(format!("Failed to wait for fence: {e}")))?;

            device
                .reset_fences(&[self.fence])
                .map_err(|e| TestError::Gpu(format!("Failed to reset fence: {e}")))?;

            device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| TestError::Gpu(format!("Failed to reset command buffer: {e}")))?;
        }

        // Read back image data
        let data = self
            .pipeline
            .read_output()
            .map_err(|e| TestError::Gpu(e.to_string()))?;

        // Clean up GPU DAG
        gpu_dag
            .destroy(&mut allocator)
            .map_err(|e| TestError::Gpu(e.to_string()))?;

        // Create image from raw data
        ImageBuffer::from_raw(self.width, self.height, data)
            .ok_or_else(|| TestError::Gpu("Failed to create image from raw data".to_string()))
    }

    /// Get the output dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl Drop for HeadlessRenderer {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe {
            let _ = device.device_wait_idle();
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.command_pool, None);
            // Pipeline cleanup requires allocator, which is complex to handle in Drop
            // For now, the pipeline resources will be cleaned up with the context
        }
    }
}

/// Visual regression test runner.
///
/// Compares rendered images against baseline images and reports differences.
pub struct VisualRegressionTest {
    config: VisualTestConfig,
    renderer: HeadlessRenderer,
}

impl VisualRegressionTest {
    /// Create a new visual regression test runner.
    ///
    /// # Arguments
    /// * `config` - Test configuration
    pub fn new(config: VisualTestConfig) -> Result<Self> {
        let renderer = HeadlessRenderer::new(256, 256)?;
        Ok(Self { config, renderer })
    }

    /// Create with custom dimensions.
    pub fn with_dimensions(config: VisualTestConfig, width: u32, height: u32) -> Result<Self> {
        let renderer = HeadlessRenderer::new(width, height)?;
        Ok(Self { config, renderer })
    }

    /// Run a test case against an SVO.
    ///
    /// # Arguments
    /// * `name` - Test case name (used for baseline filename)
    /// * `svo` - The SVO to render
    /// * `camera` - Camera configuration
    pub fn run_test(&mut self, name: &str, svo: &SparseVoxelOctree, camera: &Camera) -> Result<()> {
        let image = self.renderer.render_svo(svo, camera)?;
        self.compare_and_save(name, &image)
    }

    /// Run a test case against a DAG.
    pub fn run_test_dag(&mut self, name: &str, dag: &SvoDag, camera: &Camera) -> Result<()> {
        let image = self.renderer.render_dag(dag, camera)?;
        self.compare_and_save(name, &image)
    }

    fn compare_and_save(&self, name: &str, image: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> Result<()> {
        // Ensure directories exist
        std::fs::create_dir_all(&self.config.baseline_dir)?;
        std::fs::create_dir_all(&self.config.output_dir)?;

        let baseline_path = format!("{}/{}.png", self.config.baseline_dir, name);
        let output_path = format!("{}/{}.png", self.config.output_dir, name);

        // Save current output
        image.save(&output_path)?;

        // Compare with baseline (if exists)
        if Path::new(&baseline_path).exists() {
            let baseline = image::open(&baseline_path)
                .map_err(|e| TestError::Io(std::io::Error::other(e)))?
                .to_rgba8();

            let diff = self.compare_images(&baseline, image)?;
            if diff > self.config.threshold {
                // Save diff image
                let diff_path = format!("{}/{}_diff.png", self.config.output_dir, name);
                let diff_image = self.create_diff_image(&baseline, image);
                diff_image.save(&diff_path)?;

                return Err(TestError::ImageComparison(format!(
                    "Image difference {:.4} exceeds threshold {:.4} (see {})",
                    diff, self.config.threshold, diff_path
                )));
            }
        } else {
            // Save as new baseline
            image.save(&baseline_path)?;
            tracing::info!("Created new baseline: {}", baseline_path);
        }

        Ok(())
    }

    /// Compare two images and return the normalized difference (0.0-1.0).
    fn compare_images(
        &self,
        a: &ImageBuffer<Rgba<u8>, Vec<u8>>,
        b: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    ) -> Result<f64> {
        if a.dimensions() != b.dimensions() {
            return Err(TestError::ImageComparison(format!(
                "Image dimensions don't match: {:?} vs {:?}",
                a.dimensions(),
                b.dimensions()
            )));
        }

        let total_diff: u64 = a
            .pixels()
            .zip(b.pixels())
            .map(|(pa, pb)| {
                let diff_r = (pa[0] as i32 - pb[0] as i32).unsigned_abs() as u64;
                let diff_g = (pa[1] as i32 - pb[1] as i32).unsigned_abs() as u64;
                let diff_b = (pa[2] as i32 - pb[2] as i32).unsigned_abs() as u64;
                diff_r + diff_g + diff_b
            })
            .sum();

        let max_diff = (a.width() as u64 * a.height() as u64 * 3 * 255) as f64;
        Ok(total_diff as f64 / max_diff)
    }

    /// Create a visual diff image highlighting differences.
    fn create_diff_image(
        &self,
        a: &ImageBuffer<Rgba<u8>, Vec<u8>>,
        b: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    ) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
        let (width, height) = a.dimensions();
        let mut diff = ImageBuffer::new(width, height);

        for (x, y, pixel) in diff.enumerate_pixels_mut() {
            let pa = a.get_pixel(x, y);
            let pb = b.get_pixel(x, y);

            let diff_r = (pa[0] as i32 - pb[0] as i32).unsigned_abs() as u8;
            let diff_g = (pa[1] as i32 - pb[1] as i32).unsigned_abs() as u8;
            let diff_b = (pa[2] as i32 - pb[2] as i32).unsigned_abs() as u8;

            // Highlight differences in red
            let max_diff = diff_r.max(diff_g).max(diff_b);
            if max_diff > 10 {
                *pixel = Rgba([255, 0, 0, 255]);
            } else {
                // Show original image dimmed
                *pixel = Rgba([pa[0] / 2, pa[1] / 2, pa[2] / 2, 255]);
            }
        }

        diff
    }
}

/// Create a test camera looking at the center of an octree.
pub fn create_test_camera(octree_size: u32, distance_factor: f32) -> Camera {
    let center = octree_size as f32 / 2.0;
    let distance = octree_size as f32 * distance_factor;

    Camera {
        position: Vec3::new(center, center, center + distance),
        direction: Vec3::NEG_Z,
        up: Vec3::Y,
        fov: std::f32::consts::FRAC_PI_4,
        aspect: 1.0, // Square for testing
        near: 0.1,
        far: 1000.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxelicous_core::types::BlockId;

    // These tests require a GPU and will be skipped in CI without GPU support

    #[test]
    #[ignore = "Requires GPU hardware"]
    fn headless_renderer_creation() {
        let _renderer = HeadlessRenderer::new(256, 256).unwrap();
    }

    #[test]
    #[ignore = "Requires GPU hardware"]
    fn render_empty_octree() {
        let mut renderer = HeadlessRenderer::new(256, 256).unwrap();
        let svo = SparseVoxelOctree::new(5);
        let camera = create_test_camera(32, 1.5);

        let image = renderer.render_svo(&svo, &camera).unwrap();
        assert_eq!(image.dimensions(), (256, 256));
    }

    #[test]
    #[ignore = "Requires GPU hardware"]
    fn render_single_voxel() {
        let mut renderer = HeadlessRenderer::new(256, 256).unwrap();
        let mut svo = SparseVoxelOctree::new(5);
        svo.set(16, 16, 16, BlockId::STONE);

        let camera = create_test_camera(32, 2.0);
        let image = renderer.render_svo(&svo, &camera).unwrap();
        assert_eq!(image.dimensions(), (256, 256));
    }
}
