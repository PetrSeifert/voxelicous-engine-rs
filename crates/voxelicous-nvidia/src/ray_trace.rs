//! Hardware ray tracing pipeline for SVO-DAG rendering.
//!
//! This module provides a ray tracing pipeline that uses Vulkan's
//! `VK_KHR_ray_tracing_pipeline` extension to accelerate voxel rendering.

use ash::vk;
use gpu_allocator::MemoryLocation;
use voxelicous_gpu::descriptors::{DescriptorPool, DescriptorSetLayoutBuilder};
use voxelicous_gpu::memory::{GpuAllocator, GpuBuffer, GpuImage};
use voxelicous_gpu::{GpuError, Result};
use voxelicous_render::camera::CameraUniforms;
use voxelicous_render::svo_upload::GpuSvoDag;

use crate::acceleration::SceneAccelerationStructure;
use crate::sbt::ShaderBindingTable;

/// Push constants for ray tracing (matches compute path layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RayTracePushConstants {
    /// Screen dimensions [width, height].
    pub screen_size: [u32; 2],
    /// Maximum ray traversal steps (safety limit).
    pub max_steps: u32,
    /// Padding for alignment.
    pub _padding: u32,
    /// Device address of SVO-DAG node buffer.
    pub node_buffer_address: u64,
    /// Root node index (0 = empty octree).
    pub root_index: u32,
    /// Octree depth (2^depth = octree size).
    pub octree_depth: u32,
}

impl RayTracePushConstants {
    /// Size of push constants in bytes.
    pub const SIZE: u32 = std::mem::size_of::<Self>() as u32;
}

/// Hardware ray tracing pipeline for SVO-DAG rendering.
///
/// Uses `VK_KHR_ray_tracing_pipeline` with a custom intersection shader
/// to traverse the SVO-DAG structure directly on GPU hardware.
pub struct RayTracePipeline {
    /// Ray tracing pipeline handle.
    pipeline: vk::Pipeline,
    /// Pipeline layout.
    layout: vk::PipelineLayout,
    /// Descriptor set layout.
    descriptor_set_layout: vk::DescriptorSetLayout,
    /// Descriptor pool.
    descriptor_pool: DescriptorPool,
    /// Allocated descriptor set.
    descriptor_set: vk::DescriptorSet,

    /// Shader binding table.
    sbt: ShaderBindingTable,
    /// Scene acceleration structures (BLAS + TLAS).
    scene_as: SceneAccelerationStructure,

    /// Camera uniform buffer.
    camera_buffer: GpuBuffer,
    /// Output storage image.
    output_image: GpuImage,
    /// Image view for the output image.
    output_image_view: vk::ImageView,
    /// Readback buffer for CPU access.
    readback_buffer: GpuBuffer,

    /// Output width in pixels.
    width: u32,
    /// Output height in pixels.
    height: u32,
}

impl RayTracePipeline {
    /// Create a new ray tracing pipeline.
    ///
    /// # Arguments
    /// * `device` - Vulkan device handle
    /// * `instance` - Vulkan instance handle
    /// * `physical_device` - Physical device handle
    /// * `allocator` - GPU memory allocator
    /// * `width` - Output image width
    /// * `height` - Output image height
    /// * `octree_depth` - Depth of the SVO-DAG octree
    ///
    /// # Safety
    /// - Device must support ray tracing extensions.
    /// - All handles must be valid.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        device: &ash::Device,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        allocator: &mut GpuAllocator,
        width: u32,
        height: u32,
        octree_depth: u32,
    ) -> Result<Self> {
        // Create extension loaders
        let as_loader = ash::khr::acceleration_structure::Device::new(instance, device);
        let rt_loader = ash::khr::ray_tracing_pipeline::Device::new(instance, device);

        // Query RT pipeline properties
        let mut rt_properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut properties2 =
            vk::PhysicalDeviceProperties2::default().push_next(&mut rt_properties);
        instance.get_physical_device_properties2(physical_device, &mut properties2);

        // 1. Create scene acceleration structures
        let scene_as =
            SceneAccelerationStructure::new(device, allocator, &as_loader, octree_depth)?;

        // 2. Create descriptor set layout
        // Binding 0: Output storage image
        // Binding 1: TLAS (acceleration structure)
        // Binding 2: Camera uniform buffer
        let descriptor_set_layout = DescriptorSetLayoutBuilder::new()
            .storage_image(0, vk::ShaderStageFlags::RAYGEN_KHR)
            .acceleration_structure(1, vk::ShaderStageFlags::RAYGEN_KHR)
            .uniform_buffer(2, vk::ShaderStageFlags::RAYGEN_KHR)
            .build(device)?;

        // 3. Create push constant range
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(
                vk::ShaderStageFlags::RAYGEN_KHR
                    | vk::ShaderStageFlags::INTERSECTION_KHR
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            )
            .offset(0)
            .size(RayTracePushConstants::SIZE);

        // 4. Create pipeline layout
        let layouts = [descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));

        let layout = device.create_pipeline_layout(&layout_info, None)?;

        // 5. Load and create shader modules
        let raygen_code = voxelicous_shaders::ray_trace_svo_raygen_shader();
        let miss_code = voxelicous_shaders::ray_trace_svo_miss_shader();
        let intersection_code = voxelicous_shaders::ray_trace_svo_intersection_shader();
        let closest_hit_code = voxelicous_shaders::ray_trace_svo_closest_hit_shader();

        let raygen_module = Self::create_shader_module(device, raygen_code)?;
        let miss_module = Self::create_shader_module(device, miss_code)?;
        let intersection_module = Self::create_shader_module(device, intersection_code)?;
        let closest_hit_module = Self::create_shader_module(device, closest_hit_code)?;

        let entry_point = c"main";

        // 6. Define shader stages
        let shader_stages = [
            // 0: Ray generation
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                .module(raygen_module)
                .name(entry_point),
            // 1: Miss
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MISS_KHR)
                .module(miss_module)
                .name(entry_point),
            // 2: Intersection
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::INTERSECTION_KHR)
                .module(intersection_module)
                .name(entry_point),
            // 3: Closest-hit
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                .module(closest_hit_module)
                .name(entry_point),
        ];

        // 7. Define shader groups
        let shader_groups = [
            // Group 0: Ray generation (general)
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(0)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            // Group 1: Miss (general)
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(1)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            // Group 2: Hit group (procedural with intersection + closest-hit)
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(3)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(2),
        ];

        // 8. Create ray tracing pipeline
        let pipeline_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&shader_stages)
            .groups(&shader_groups)
            .max_pipeline_ray_recursion_depth(1)
            .layout(layout);

        let pipelines = rt_loader
            .create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
            .map_err(|e| GpuError::PipelineCreation(format!("Ray tracing pipeline: {e:?}")))?;

        let pipeline = pipelines[0];

        // Destroy shader modules (no longer needed)
        device.destroy_shader_module(raygen_module, None);
        device.destroy_shader_module(miss_module, None);
        device.destroy_shader_module(intersection_module, None);
        device.destroy_shader_module(closest_hit_module, None);

        // 9. Create shader binding table
        let sbt = ShaderBindingTable::new(device, &rt_loader, allocator, pipeline, &rt_properties)?;

        // 10. Create camera uniform buffer
        let camera_buffer = allocator.create_buffer(
            std::mem::size_of::<CameraUniforms>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "rt_camera_uniforms",
        )?;

        // 11. Create output storage image
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
            allocator.create_image(&image_info, MemoryLocation::GpuOnly, "rt_output")?;

        // 12. Create image view
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

        // 13. Create readback buffer
        let readback_buffer = allocator.create_buffer(
            (width * height * 4) as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
            "rt_readback",
        )?;

        // 14. Create descriptor pool
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
        ];

        let descriptor_pool = DescriptorPool::new(device, 1, &pool_sizes)?;

        // 15. Allocate descriptor set
        let descriptor_sets = descriptor_pool.allocate(device, &[descriptor_set_layout])?;
        let descriptor_set = descriptor_sets[0];

        // 16. Write descriptor set
        let image_info_desc = vk::DescriptorImageInfo::default()
            .image_view(output_image_view)
            .image_layout(vk::ImageLayout::GENERAL);

        let as_handle = scene_as.tlas().acceleration_structure;
        let mut as_write_info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(std::slice::from_ref(&as_handle));

        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(camera_buffer.buffer)
            .offset(0)
            .range(std::mem::size_of::<CameraUniforms>() as u64);

        let writes = [
            // Binding 0: Output image
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&image_info_desc)),
            // Binding 1: TLAS
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .push_next(&mut as_write_info),
            // Binding 2: Camera buffer
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info)),
        ];

        device.update_descriptor_sets(&writes, &[]);

        Ok(Self {
            pipeline,
            layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            sbt,
            scene_as,
            camera_buffer,
            output_image,
            output_image_view,
            readback_buffer,
            width,
            height,
        })
    }

    /// Create a shader module from SPIR-V code.
    unsafe fn create_shader_module(device: &ash::Device, code: &[u32]) -> Result<vk::ShaderModule> {
        let create_info = vk::ShaderModuleCreateInfo::default().code(code);
        device
            .create_shader_module(&create_info, None)
            .map_err(|e| {
                GpuError::ShaderCompilation(format!("Failed to create shader module: {e}"))
            })
    }

    /// Build acceleration structures. Call once after creation.
    ///
    /// # Safety
    /// - All handles must be valid.
    /// - Command buffer must be in recording state.
    pub unsafe fn build_acceleration_structures(
        &self,
        device: &ash::Device,
        instance: &ash::Instance,
        cmd: vk::CommandBuffer,
    ) {
        let as_loader = ash::khr::acceleration_structure::Device::new(instance, device);
        self.scene_as.record_build(device, &as_loader, cmd);
    }

    /// Record ray trace dispatch commands into a command buffer.
    ///
    /// # Arguments
    /// * `device` - Vulkan device handle
    /// * `instance` - Vulkan instance handle
    /// * `cmd` - Command buffer (must be in recording state)
    /// * `camera` - Camera uniforms to use
    /// * `svo` - GPU-resident SVO-DAG to render
    /// * `max_steps` - Maximum ray traversal steps
    ///
    /// # Safety
    /// The command buffer must be in recording state.
    pub unsafe fn record(
        &self,
        device: &ash::Device,
        instance: &ash::Instance,
        cmd: vk::CommandBuffer,
        camera: &CameraUniforms,
        svo: &GpuSvoDag,
        max_steps: u32,
    ) -> Result<()> {
        let rt_loader = ash::khr::ray_tracing_pipeline::Device::new(instance, device);

        // Update camera buffer
        self.camera_buffer.write(std::slice::from_ref(camera))?;

        // Transition output image to GENERAL layout
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
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

        // Bind pipeline and descriptors
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::RAY_TRACING_KHR, self.pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            self.layout,
            0,
            &[self.descriptor_set],
            &[],
        );

        // Push constants
        let push_constants = RayTracePushConstants {
            screen_size: [self.width, self.height],
            max_steps,
            _padding: 0,
            node_buffer_address: svo.device_address,
            root_index: svo.root_index,
            octree_depth: svo.depth,
        };

        device.cmd_push_constants(
            cmd,
            self.layout,
            vk::ShaderStageFlags::RAYGEN_KHR
                | vk::ShaderStageFlags::INTERSECTION_KHR
                | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            0,
            bytemuck::bytes_of(&push_constants),
        );

        // Trace rays
        rt_loader.cmd_trace_rays(
            cmd,
            &self.sbt.raygen_region,
            &self.sbt.miss_region,
            &self.sbt.hit_region,
            &self.sbt.callable_region,
            self.width,
            self.height,
            1,
        );

        Ok(())
    }

    /// Record commands to copy the output image to the readback buffer.
    ///
    /// # Safety
    /// The command buffer must be in recording state.
    pub unsafe fn record_readback(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        // Transition output image for transfer
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
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

    /// Get a reference to the scene acceleration structure.
    pub fn scene_as(&self) -> &SceneAccelerationStructure {
        &self.scene_as
    }

    /// Destroy all GPU resources.
    ///
    /// # Safety
    /// - Device must be valid.
    /// - All resources must not be in use.
    pub unsafe fn destroy(
        mut self,
        device: &ash::Device,
        instance: &ash::Instance,
        allocator: &mut GpuAllocator,
    ) -> Result<()> {
        let as_loader = ash::khr::acceleration_structure::Device::new(instance, device);

        device.destroy_image_view(self.output_image_view, None);
        allocator.free_image(&mut self.output_image)?;
        allocator.free_buffer(&mut self.camera_buffer)?;
        allocator.free_buffer(&mut self.readback_buffer)?;

        self.sbt.destroy(allocator)?;
        self.scene_as.destroy(&as_loader, allocator)?;

        self.descriptor_pool.destroy(device);
        device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.layout, None);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_constants_size() {
        assert_eq!(RayTracePushConstants::SIZE, 32);
    }

    #[test]
    fn push_constants_layout() {
        // Verify memory layout matches shader expectations
        let pc = RayTracePushConstants {
            screen_size: [1920, 1080],
            max_steps: 256,
            _padding: 0,
            node_buffer_address: 0x1234_5678_9ABC_DEF0,
            root_index: 1,
            octree_depth: 5,
        };

        let bytes = bytemuck::bytes_of(&pc);
        assert_eq!(bytes.len(), 32);

        // Check screen_size at offset 0
        assert_eq!(&bytes[0..4], &1920u32.to_ne_bytes());
        assert_eq!(&bytes[4..8], &1080u32.to_ne_bytes());

        // Check max_steps at offset 8
        assert_eq!(&bytes[8..12], &256u32.to_ne_bytes());

        // Check node_buffer_address at offset 16
        assert_eq!(&bytes[16..24], &0x1234_5678_9ABC_DEF0u64.to_ne_bytes());

        // Check root_index at offset 24
        assert_eq!(&bytes[24..28], &1u32.to_ne_bytes());

        // Check octree_depth at offset 28
        assert_eq!(&bytes[28..32], &5u32.to_ne_bytes());
    }
}
