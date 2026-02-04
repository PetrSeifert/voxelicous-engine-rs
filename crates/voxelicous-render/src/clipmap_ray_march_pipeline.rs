//! Clipmap ray marching compute pipeline.
//!
//! Uses the `ray_march_clipmap.comp` shader to render clipmap voxel data.

use ash::vk;
use gpu_allocator::MemoryLocation;
use voxelicous_gpu::descriptors::{DescriptorPool, DescriptorSetLayoutBuilder};
use voxelicous_gpu::error::{GpuError, Result};
use voxelicous_gpu::memory::{GpuAllocator, GpuBuffer, GpuImage};
use voxelicous_gpu::pipeline::ComputePipeline;

use crate::camera::CameraUniforms;
use crate::clipmap_render::{ClipmapRenderPushConstants, ClipmapRenderer};
use crate::debug::DebugMode;

/// Clipmap ray marching compute pipeline.
pub struct ClipmapRayMarchPipeline {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    camera_buffers: Vec<GpuBuffer>,
    output_image: GpuImage,
    output_image_view: vk::ImageView,
    readback_buffer: GpuBuffer,
    frames_in_flight: usize,
    width: u32,
    height: u32,
}

impl ClipmapRayMarchPipeline {
    /// Create a new clipmap ray marching pipeline.
    ///
    /// # Safety
    /// The Vulkan device must be valid.
    pub unsafe fn new(
        device: &ash::Device,
        allocator: &mut GpuAllocator,
        width: u32,
        height: u32,
        frames_in_flight: usize,
    ) -> Result<Self> {
        let descriptor_set_layout = DescriptorSetLayoutBuilder::new()
            .uniform_buffer(0, vk::ShaderStageFlags::COMPUTE)
            .storage_image(1, vk::ShaderStageFlags::COMPUTE)
            .build(device)?;

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(ClipmapRenderPushConstants::SIZE);

        let shader_code = voxelicous_shaders::ray_march_clipmap_shader();
        let pipeline = ComputePipeline::new(
            device,
            shader_code,
            &[descriptor_set_layout],
            &[push_constant_range],
        )?;

        let mut camera_buffers = Vec::with_capacity(frames_in_flight);
        for i in 0..frames_in_flight {
            let buffer = allocator.create_buffer(
                std::mem::size_of::<CameraUniforms>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
                &format!("clipmap_camera_uniforms_{i}"),
            )?;
            camera_buffers.push(buffer);
        }

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let output_image =
            allocator.create_image(&image_info, MemoryLocation::GpuOnly, "clipmap_output")?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(output_image.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let output_image_view = device
            .create_image_view(&view_info, None)
            .map_err(|e| GpuError::Other(format!("Failed to create image view: {e}")))?;

        let readback_buffer = allocator.create_buffer(
            (width * height * 4) as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
            "clipmap_readback",
        )?;

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(frames_in_flight as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(frames_in_flight as u32),
        ];

        let descriptor_pool = DescriptorPool::new(device, frames_in_flight as u32, &pool_sizes)?;
        let layouts: Vec<_> = (0..frames_in_flight)
            .map(|_| descriptor_set_layout)
            .collect();
        let descriptor_sets = descriptor_pool.allocate(device, &layouts)?;

        let image_info_desc = vk::DescriptorImageInfo::default()
            .image_view(output_image_view)
            .image_layout(vk::ImageLayout::GENERAL);

        for (i, &descriptor_set) in descriptor_sets.iter().enumerate() {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(camera_buffers[i].buffer)
                .offset(0)
                .range(std::mem::size_of::<CameraUniforms>() as u64);

            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&image_info_desc)),
            ];

            device.update_descriptor_sets(&writes, &[]);
        }

        Ok(Self {
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            camera_buffers,
            output_image,
            output_image_view,
            readback_buffer,
            frames_in_flight,
            width,
            height,
        })
    }

    /// Record clipmap ray marching dispatch commands.
    ///
    /// # Safety
    /// Command buffer must be in recording state.
    pub unsafe fn record(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        camera: &CameraUniforms,
        renderer: &ClipmapRenderer,
        max_steps: u32,
        frame_index: usize,
        debug_mode: DebugMode,
    ) -> Result<()> {
        self.camera_buffers[frame_index].write(std::slice::from_ref(camera))?;

        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .image(self.output_image.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&barrier));

        device.cmd_pipeline_barrier2(cmd, &dependency_info);

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline.pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline.layout,
            0,
            &[self.descriptor_sets[frame_index]],
            &[],
        );

        let push_constants =
            renderer.push_constants(self.width, self.height, max_steps, frame_index, debug_mode);

        device.cmd_push_constants(
            cmd,
            self.pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push_constants),
        );

        let workgroup_x = (self.width + 7) / 8;
        let workgroup_y = (self.height + 7) / 8;
        device.cmd_dispatch(cmd, workgroup_x, workgroup_y, 1);

        Ok(())
    }

    /// Record commands to copy the output image to the readback buffer.
    pub unsafe fn record_readback_from_transfer_src(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
    ) {
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D::default())
            .image_extent(vk::Extent3D {
                width: self.width,
                height: self.height,
                depth: 1,
            });

        device.cmd_copy_image_to_buffer(
            cmd,
            self.output_image.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            self.readback_buffer.buffer,
            &[region],
        );
    }

    /// Read the rendered image from the readback buffer.
    pub fn read_output(&self) -> Result<Vec<u8>> {
        let ptr = self
            .readback_buffer
            .mapped_ptr()
            .ok_or_else(|| GpuError::InvalidState("Readback buffer not mapped".to_string()))?;

        let size = (self.width * self.height * 4) as usize;
        let mut data = vec![0u8; size];

        unsafe {
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), size);
        }

        Ok(data)
    }

    /// Get output image dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Access output image.
    pub fn output_image(&self) -> &GpuImage {
        &self.output_image
    }

    /// Destroy GPU resources.
    ///
    /// # Safety
    /// The device must be idle.
    pub unsafe fn destroy(
        mut self,
        device: &ash::Device,
        allocator: &mut GpuAllocator,
    ) -> Result<()> {
        device.destroy_image_view(self.output_image_view, None);
        allocator.free_image(&mut self.output_image)?;
        for camera_buffer in &mut self.camera_buffers {
            allocator.free_buffer(camera_buffer)?;
        }
        allocator.free_buffer(&mut self.readback_buffer)?;
        self.descriptor_pool.destroy(device);
        device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        self.pipeline.destroy(device);
        Ok(())
    }
}
