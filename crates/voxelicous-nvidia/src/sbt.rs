//! Shader Binding Table (SBT) management for hardware ray tracing.
//!
//! The SBT is a GPU buffer containing shader handles organized by type:
//! - Ray generation shaders
//! - Miss shaders
//! - Hit groups (intersection + any-hit + closest-hit)
//! - Callable shaders (unused in this implementation)

use ash::vk;
use gpu_allocator::MemoryLocation;
use voxelicous_gpu::{GpuAllocator, GpuBuffer, GpuError, Result};

/// Shader Binding Table for ray tracing pipeline.
///
/// Layout:
/// - Ray Generation: 1 shader
/// - Miss: 1 shader
/// - Hit Groups: 1 hit group (intersection + closest-hit)
pub struct ShaderBindingTable {
    /// Combined SBT buffer containing all shader handles.
    pub buffer: GpuBuffer,

    /// Ray generation shader region.
    pub raygen_region: vk::StridedDeviceAddressRegionKHR,
    /// Miss shader region.
    pub miss_region: vk::StridedDeviceAddressRegionKHR,
    /// Hit group region.
    pub hit_region: vk::StridedDeviceAddressRegionKHR,
    /// Callable region (empty but required).
    pub callable_region: vk::StridedDeviceAddressRegionKHR,
}

impl ShaderBindingTable {
    /// Create a new Shader Binding Table from a ray tracing pipeline.
    ///
    /// # Arguments
    /// * `device` - Vulkan device
    /// * `rt_loader` - Ray tracing pipeline extension loader
    /// * `allocator` - GPU memory allocator
    /// * `pipeline` - Ray tracing pipeline handle
    /// * `rt_properties` - Ray tracing pipeline properties from physical device
    /// * `group_count` - Number of shader groups in the pipeline (raygen + miss + hit groups)
    ///
    /// # Safety
    /// - Device and allocator must be valid.
    /// - RT loader must be properly initialized.
    /// - Pipeline must be a valid ray tracing pipeline.
    pub unsafe fn new(
        device: &ash::Device,
        rt_loader: &ash::khr::ray_tracing_pipeline::Device,
        allocator: &mut GpuAllocator,
        pipeline: vk::Pipeline,
        rt_properties: &vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    ) -> Result<Self> {
        let handle_size = rt_properties.shader_group_handle_size;
        let handle_alignment = rt_properties.shader_group_handle_alignment;
        let base_alignment = rt_properties.shader_group_base_alignment;

        // Align handle size to handle alignment
        let aligned_handle_size = Self::align_up(handle_size, handle_alignment);

        // Group counts (hardcoded for our pipeline: 1 raygen, 1 miss, 1 hit group)
        let raygen_count = 1u32;
        let miss_count = 1u32;
        let hit_count = 1u32;
        let group_count = raygen_count + miss_count + hit_count;

        // Calculate region sizes (aligned to base alignment)
        let raygen_size = Self::align_up(aligned_handle_size * raygen_count, base_alignment);
        let miss_size = Self::align_up(aligned_handle_size * miss_count, base_alignment);
        let hit_size = Self::align_up(aligned_handle_size * hit_count, base_alignment);

        // Total buffer size
        let sbt_size = raygen_size + miss_size + hit_size;

        // Get shader group handles from pipeline
        let handle_data_size = (handle_size * group_count) as usize;
        let handles = rt_loader.get_ray_tracing_shader_group_handles(
            pipeline,
            0,
            group_count,
            handle_data_size,
        )?;

        // Create SBT buffer
        let buffer = allocator.create_buffer(
            sbt_size as u64,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
            "shader_binding_table",
        )?;

        let sbt_ptr = buffer
            .mapped_ptr()
            .ok_or_else(|| GpuError::InvalidState("SBT buffer not mapped".to_string()))?;

        // Copy handles to buffer with proper alignment
        let handle_size_usize = handle_size as usize;

        // Raygen (group 0)
        let raygen_offset = 0usize;
        std::ptr::copy_nonoverlapping(
            handles.as_ptr(),
            sbt_ptr.add(raygen_offset),
            handle_size_usize,
        );

        // Miss (group 1)
        let miss_offset = raygen_size as usize;
        std::ptr::copy_nonoverlapping(
            handles.as_ptr().add(handle_size_usize),
            sbt_ptr.add(miss_offset),
            handle_size_usize,
        );

        // Hit group (group 2)
        let hit_offset = (raygen_size + miss_size) as usize;
        std::ptr::copy_nonoverlapping(
            handles.as_ptr().add(handle_size_usize * 2),
            sbt_ptr.add(hit_offset),
            handle_size_usize,
        );

        // Get buffer device address
        let buffer_address = buffer.device_address(device);

        // Create regions
        // Note: For raygen, size MUST equal stride per Vulkan spec (VUID-vkCmdTraceRaysKHR-size-04023)
        // The buffer still uses raygen_size for layout/alignment, but the region size = stride
        let raygen_region = vk::StridedDeviceAddressRegionKHR {
            device_address: buffer_address,
            stride: aligned_handle_size as u64,
            size: aligned_handle_size as u64,
        };

        let miss_region = vk::StridedDeviceAddressRegionKHR {
            device_address: buffer_address + raygen_size as u64,
            stride: aligned_handle_size as u64,
            size: miss_size as u64,
        };

        let hit_region = vk::StridedDeviceAddressRegionKHR {
            device_address: buffer_address + raygen_size as u64 + miss_size as u64,
            stride: aligned_handle_size as u64,
            size: hit_size as u64,
        };

        // Empty callable region
        let callable_region = vk::StridedDeviceAddressRegionKHR {
            device_address: 0,
            stride: 0,
            size: 0,
        };

        Ok(Self {
            buffer,
            raygen_region,
            miss_region,
            hit_region,
            callable_region,
        })
    }

    /// Align a value up to the given alignment.
    fn align_up(value: u32, alignment: u32) -> u32 {
        (value + alignment - 1) & !(alignment - 1)
    }

    /// Destroy the SBT and free resources.
    ///
    /// # Safety
    /// - Allocator must be valid.
    /// - The SBT must not be in use.
    pub unsafe fn destroy(mut self, allocator: &mut GpuAllocator) -> Result<()> {
        allocator.free_buffer(&mut self.buffer)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_test() {
        assert_eq!(ShaderBindingTable::align_up(32, 64), 64);
        assert_eq!(ShaderBindingTable::align_up(64, 64), 64);
        assert_eq!(ShaderBindingTable::align_up(65, 64), 128);
        assert_eq!(ShaderBindingTable::align_up(1, 4), 4);
        assert_eq!(ShaderBindingTable::align_up(4, 4), 4);
    }
}
