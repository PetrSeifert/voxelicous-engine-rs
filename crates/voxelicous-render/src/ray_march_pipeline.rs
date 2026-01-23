//! Ray marching compute pipeline.
//!
//! Orchestrates the compute shader for ray marching through SVO-DAG structures.

use ash::vk;
use gpu_allocator::MemoryLocation;
use voxelicous_gpu::descriptors::{DescriptorPool, DescriptorSetLayoutBuilder};
use voxelicous_gpu::error::{GpuError, Result};
use voxelicous_gpu::memory::{GpuAllocator, GpuBuffer, GpuImage};
use voxelicous_gpu::pipeline::ComputePipeline;

use crate::camera::CameraUniforms;
use crate::ray_march::RayMarchPushConstants;
use crate::svo_upload::GpuSvoDag;

/// Ray marching compute pipeline.
///
/// Manages the GPU resources and pipeline state for ray marching
/// through an SVO-DAG structure.
pub struct RayMarchPipeline {
    /// The compute pipeline.
    pipeline: ComputePipeline,
    /// Descriptor set layout.
    descriptor_set_layout: vk::DescriptorSetLayout,
    /// Descriptor pool.
    descriptor_pool: DescriptorPool,
    /// Per-frame allocated descriptor sets.
    descriptor_sets: Vec<vk::DescriptorSet>,

    /// Per-frame camera uniform buffers.
    camera_buffers: Vec<GpuBuffer>,
    /// Output storage image.
    output_image: GpuImage,
    /// Image view for the output image.
    output_image_view: vk::ImageView,
    /// Readback buffer for CPU access.
    readback_buffer: GpuBuffer,

    /// Number of frames in flight.
    #[allow(dead_code)]
    frames_in_flight: usize,
    /// Output width in pixels.
    width: u32,
    /// Output height in pixels.
    height: u32,
}

impl RayMarchPipeline {
    /// Create a new ray marching pipeline.
    ///
    /// # Arguments
    /// * `device` - Vulkan device handle
    /// * `allocator` - GPU memory allocator
    /// * `width` - Output image width
    /// * `height` - Output image height
    /// * `frames_in_flight` - Number of frames that can be in flight simultaneously
    ///
    /// # Safety
    /// The device must be valid and support compute shaders.
    pub unsafe fn new(
        device: &ash::Device,
        allocator: &mut GpuAllocator,
        width: u32,
        height: u32,
        frames_in_flight: usize,
    ) -> Result<Self> {
        // 1. Create descriptor set layout
        // Binding 0: Camera uniform buffer
        // Binding 1: Output storage image
        let descriptor_set_layout = DescriptorSetLayoutBuilder::new()
            .uniform_buffer(0, vk::ShaderStageFlags::COMPUTE)
            .storage_image(1, vk::ShaderStageFlags::COMPUTE)
            .build(device)?;

        // 2. Create push constant range
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(RayMarchPushConstants::SIZE);

        // 3. Create compute pipeline
        let shader_code = voxelicous_shaders::ray_march_svo_shader();
        let pipeline = ComputePipeline::new(
            device,
            shader_code,
            &[descriptor_set_layout],
            &[push_constant_range],
        )?;

        // 4. Create per-frame camera uniform buffers
        let mut camera_buffers = Vec::with_capacity(frames_in_flight);
        for i in 0..frames_in_flight {
            let buffer = allocator.create_buffer(
                std::mem::size_of::<CameraUniforms>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
                &format!("camera_uniforms_{i}"),
            )?;
            camera_buffers.push(buffer);
        }

        // 5. Create output storage image
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
            allocator.create_image(&image_info, MemoryLocation::GpuOnly, "ray_march_output")?;

        // 6. Create image view
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

        // 7. Create readback buffer for CPU access
        let readback_buffer = allocator.create_buffer(
            (width * height * 4) as u64, // RGBA8
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
            "ray_march_readback",
        )?;

        // 8. Create descriptor pool (one set per frame in flight)
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(frames_in_flight as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(frames_in_flight as u32),
        ];

        let descriptor_pool = DescriptorPool::new(device, frames_in_flight as u32, &pool_sizes)?;

        // 9. Allocate per-frame descriptor sets
        let layouts: Vec<_> = (0..frames_in_flight)
            .map(|_| descriptor_set_layout)
            .collect();
        let descriptor_sets = descriptor_pool.allocate(device, &layouts)?;

        // 10. Write per-frame descriptor sets
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

    /// Record ray march dispatch commands into a command buffer.
    ///
    /// # Arguments
    /// * `device` - Vulkan device handle
    /// * `cmd` - Command buffer (must be in recording state)
    /// * `camera` - Camera uniforms to use
    /// * `svo` - GPU-resident SVO-DAG to render
    /// * `max_steps` - Maximum ray traversal steps
    /// * `frame_index` - Current frame index in the ring buffer
    ///
    /// # Safety
    /// The command buffer must be in recording state.
    pub unsafe fn record(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        camera: &CameraUniforms,
        svo: &GpuSvoDag,
        max_steps: u32,
        frame_index: usize,
    ) -> Result<()> {
        // Update this frame's camera buffer
        self.camera_buffers[frame_index].write(std::slice::from_ref(camera))?;

        // Transition output image to GENERAL layout for compute shader
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

        // Bind pipeline and this frame's descriptor set
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline.pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline.layout,
            0,
            &[self.descriptor_sets[frame_index]],
            &[],
        );

        // Push constants
        let push_constants = RayMarchPushConstants {
            screen_size: [self.width, self.height],
            max_steps,
            _padding: 0,
            node_buffer_address: svo.device_address,
            root_index: svo.root_index,
            octree_depth: svo.depth,
        };

        device.cmd_push_constants(
            cmd,
            self.pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push_constants),
        );

        // Dispatch compute
        let workgroup_x = (self.width + 7) / 8;
        let workgroup_y = (self.height + 7) / 8;
        device.cmd_dispatch(cmd, workgroup_x, workgroup_y, 1);

        Ok(())
    }

    /// Record commands to copy the output image to the readback buffer.
    ///
    /// Call this after `record()` to prepare for CPU readback.
    ///
    /// # Safety
    /// The command buffer must be in recording state.
    pub unsafe fn record_readback(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        // Transition output image for transfer
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
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

        // Copy image to buffer
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

    /// Record readback commands when the image is already in TRANSFER_SRC_OPTIMAL layout.
    ///
    /// Use this variant when the image has already been transitioned (e.g., after a blit).
    /// This avoids the Vulkan validation error from mismatched layout transitions.
    ///
    /// # Safety
    /// The command buffer must be in recording state.
    /// The output image must already be in TRANSFER_SRC_OPTIMAL layout.
    pub unsafe fn record_readback_from_transfer_src(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
    ) {
        // Copy image to buffer (image is already in TRANSFER_SRC_OPTIMAL)
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
    ///
    /// Call this after submitting commands and waiting for completion.
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

    /// Get the output image dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get a reference to the output image.
    pub fn output_image(&self) -> &GpuImage {
        &self.output_image
    }

    /// Destroy all GPU resources.
    ///
    /// # Safety
    /// The device must be valid and all resources must not be in use.
    pub unsafe fn destroy(
        mut self,
        device: &ash::Device,
        allocator: &mut GpuAllocator,
    ) -> Result<()> {
        device.destroy_image_view(self.output_image_view, None);
        allocator.free_image(&mut self.output_image)?;
        // Free all per-frame camera buffers
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
