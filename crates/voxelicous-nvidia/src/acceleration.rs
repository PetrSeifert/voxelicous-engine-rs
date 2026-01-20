//! Acceleration structure management for hardware ray tracing.
//!
//! This module provides BLAS (Bottom-Level Acceleration Structure) and
//! TLAS (Top-Level Acceleration Structure) management for Vulkan ray tracing.

use ash::vk;
use gpu_allocator::MemoryLocation;
use voxelicous_gpu::{GpuAllocator, GpuBuffer, GpuError, Result};

/// AABB positions for procedural geometry (24 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AabbPositions {
    pub min_x: f32,
    pub min_y: f32,
    pub min_z: f32,
    pub max_x: f32,
    pub max_y: f32,
    pub max_z: f32,
}

impl AabbPositions {
    /// Create AABB for an octree with the given size.
    pub fn from_octree_size(size: f32) -> Self {
        Self {
            min_x: 0.0,
            min_y: 0.0,
            min_z: 0.0,
            max_x: size,
            max_y: size,
            max_z: size,
        }
    }
}

/// Bottom-Level Acceleration Structure for procedural geometry.
///
/// Uses a single procedural AABB to represent the entire octree bounds.
/// The actual voxel intersection is handled by a custom intersection shader.
pub struct ProceduralBlas {
    /// Vulkan acceleration structure handle.
    pub acceleration_structure: vk::AccelerationStructureKHR,
    /// Backing buffer for the acceleration structure.
    pub buffer: GpuBuffer,
    /// AABB buffer containing the procedural geometry bounds.
    pub aabb_buffer: GpuBuffer,
    /// Device address of the acceleration structure.
    pub device_address: vk::DeviceAddress,
}

impl ProceduralBlas {
    /// Create a new BLAS with a single procedural AABB.
    ///
    /// # Safety
    /// - Device and allocator must be valid.
    /// - AS loader must be properly initialized.
    pub unsafe fn new(
        device: &ash::Device,
        allocator: &mut GpuAllocator,
        as_loader: &ash::khr::acceleration_structure::Device,
        octree_size: f32,
    ) -> Result<Self> {
        // Create AABB buffer
        let aabb = AabbPositions::from_octree_size(octree_size);
        let aabb_buffer = allocator.create_buffer(
            std::mem::size_of::<AabbPositions>() as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
            "blas_aabb_buffer",
        )?;
        aabb_buffer.write(&[aabb])?;

        let aabb_buffer_address = aabb_buffer.device_address(device);

        // Define geometry
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::AABBS)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                aabbs: vk::AccelerationStructureGeometryAabbsDataKHR::default()
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: aabb_buffer_address,
                    })
                    .stride(std::mem::size_of::<AabbPositions>() as u64),
            });

        // Query build sizes
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&geometry));

        let primitive_count = 1u32;
        let mut build_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        as_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[primitive_count],
            &mut build_sizes,
        );

        // Create AS buffer
        let buffer = allocator.create_buffer(
            build_sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
            "blas_buffer",
        )?;

        // Create acceleration structure
        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(buffer.buffer)
            .offset(0)
            .size(build_sizes.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

        let acceleration_structure = as_loader.create_acceleration_structure(&create_info, None)?;

        // Get device address
        let address_info = vk::AccelerationStructureDeviceAddressInfoKHR::default()
            .acceleration_structure(acceleration_structure);
        let device_address = as_loader.get_acceleration_structure_device_address(&address_info);

        Ok(Self {
            acceleration_structure,
            buffer,
            aabb_buffer,
            device_address,
        })
    }

    /// Get the build sizes for this BLAS.
    ///
    /// # Safety
    /// The AS loader must be valid.
    pub unsafe fn get_build_sizes(
        &self,
        device: &ash::Device,
        as_loader: &ash::khr::acceleration_structure::Device,
    ) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
        let aabb_buffer_address = self.aabb_buffer.device_address(device);

        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::AABBS)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                aabbs: vk::AccelerationStructureGeometryAabbsDataKHR::default()
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: aabb_buffer_address,
                    })
                    .stride(std::mem::size_of::<AabbPositions>() as u64),
            });

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&geometry));

        let mut build_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        as_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[1],
            &mut build_sizes,
        );
        build_sizes
    }

    /// Record build commands for this BLAS.
    ///
    /// # Safety
    /// - All handles must be valid.
    /// - The command buffer must be in recording state.
    /// - The scratch buffer must be large enough.
    pub unsafe fn record_build(
        &self,
        device: &ash::Device,
        as_loader: &ash::khr::acceleration_structure::Device,
        cmd: vk::CommandBuffer,
        scratch_buffer: &GpuBuffer,
    ) {
        let aabb_buffer_address = self.aabb_buffer.device_address(device);
        let scratch_address = scratch_buffer.device_address(device);

        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::AABBS)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                aabbs: vk::AccelerationStructureGeometryAabbsDataKHR::default()
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: aabb_buffer_address,
                    })
                    .stride(std::mem::size_of::<AabbPositions>() as u64),
            });

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(self.acceleration_structure)
            .geometries(std::slice::from_ref(&geometry))
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_address,
            });

        let build_range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(1)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);

        as_loader.cmd_build_acceleration_structures(
            cmd,
            &[build_info],
            &[std::slice::from_ref(&build_range)],
        );
    }

    /// Destroy the BLAS and free resources.
    ///
    /// # Safety
    /// - All handles must be valid.
    /// - The BLAS must not be in use.
    pub unsafe fn destroy(
        mut self,
        as_loader: &ash::khr::acceleration_structure::Device,
        allocator: &mut GpuAllocator,
    ) -> Result<()> {
        as_loader.destroy_acceleration_structure(self.acceleration_structure, None);
        allocator.free_buffer(&mut self.buffer)?;
        allocator.free_buffer(&mut self.aabb_buffer)?;
        Ok(())
    }
}

/// Top-Level Acceleration Structure.
///
/// Contains a single instance referencing the BLAS.
pub struct Tlas {
    /// Vulkan acceleration structure handle.
    pub acceleration_structure: vk::AccelerationStructureKHR,
    /// Backing buffer for the acceleration structure.
    pub buffer: GpuBuffer,
    /// Instance buffer containing BLAS references.
    pub instance_buffer: GpuBuffer,
    /// Device address of the acceleration structure.
    pub device_address: vk::DeviceAddress,
}

impl Tlas {
    /// Create a new TLAS with a single instance referencing the BLAS.
    ///
    /// # Safety
    /// - Device and allocator must be valid.
    /// - AS loader must be properly initialized.
    /// - BLAS must be built before building TLAS.
    pub unsafe fn new(
        device: &ash::Device,
        allocator: &mut GpuAllocator,
        as_loader: &ash::khr::acceleration_structure::Device,
        blas: &ProceduralBlas,
    ) -> Result<Self> {
        // Create instance data with identity transform (row-major 3x4 matrix)
        let transform = vk::TransformMatrixKHR {
            matrix: [
                1.0, 0.0, 0.0, 0.0, // row 0: scale_x=1, no rotation, translate_x=0
                0.0, 1.0, 0.0, 0.0, // row 1: no rotation, scale_y=1, translate_y=0
                0.0, 0.0, 1.0, 0.0, // row 2: no rotation, scale_z=1, translate_z=0
            ],
        };

        let instance = vk::AccelerationStructureInstanceKHR {
            transform,
            instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: blas.device_address,
            },
        };

        // Create instance buffer
        let instance_buffer = allocator.create_buffer(
            std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::CpuToGpu,
            "tlas_instance_buffer",
        )?;

        // Write instance data
        let ptr = instance_buffer
            .mapped_ptr()
            .ok_or_else(|| GpuError::InvalidState("Instance buffer not mapped".to_string()))?;
        std::ptr::copy_nonoverlapping(
            &instance as *const _ as *const u8,
            ptr,
            std::mem::size_of::<vk::AccelerationStructureInstanceKHR>(),
        );

        let instance_buffer_address = instance_buffer.device_address(device);

        // Define geometry
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: instance_buffer_address,
                    }),
            });

        // Query build sizes
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&geometry));

        let instance_count = 1u32;
        let mut build_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        as_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[instance_count],
            &mut build_sizes,
        );

        // Create AS buffer
        let buffer = allocator.create_buffer(
            build_sizes.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
            "tlas_buffer",
        )?;

        // Create acceleration structure
        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(buffer.buffer)
            .offset(0)
            .size(build_sizes.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

        let acceleration_structure = as_loader.create_acceleration_structure(&create_info, None)?;

        // Get device address
        let address_info = vk::AccelerationStructureDeviceAddressInfoKHR::default()
            .acceleration_structure(acceleration_structure);
        let device_address = as_loader.get_acceleration_structure_device_address(&address_info);

        Ok(Self {
            acceleration_structure,
            buffer,
            instance_buffer,
            device_address,
        })
    }

    /// Get the build sizes for this TLAS.
    ///
    /// # Safety
    /// The AS loader must be valid.
    pub unsafe fn get_build_sizes(
        &self,
        device: &ash::Device,
        as_loader: &ash::khr::acceleration_structure::Device,
    ) -> vk::AccelerationStructureBuildSizesInfoKHR<'static> {
        let instance_buffer_address = self.instance_buffer.device_address(device);

        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: instance_buffer_address,
                    }),
            });

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&geometry));

        let mut build_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        as_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build_info,
            &[1],
            &mut build_sizes,
        );
        build_sizes
    }

    /// Record build commands for this TLAS.
    ///
    /// # Safety
    /// - All handles must be valid.
    /// - The command buffer must be in recording state.
    /// - The scratch buffer must be large enough.
    /// - The BLAS must be built before building TLAS.
    pub unsafe fn record_build(
        &self,
        device: &ash::Device,
        as_loader: &ash::khr::acceleration_structure::Device,
        cmd: vk::CommandBuffer,
        scratch_buffer: &GpuBuffer,
    ) {
        let instance_buffer_address = self.instance_buffer.device_address(device);
        let scratch_address = scratch_buffer.device_address(device);

        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: instance_buffer_address,
                    }),
            });

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(self.acceleration_structure)
            .geometries(std::slice::from_ref(&geometry))
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_address,
            });

        let build_range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(1)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);

        as_loader.cmd_build_acceleration_structures(
            cmd,
            &[build_info],
            &[std::slice::from_ref(&build_range)],
        );
    }

    /// Destroy the TLAS and free resources.
    ///
    /// # Safety
    /// - All handles must be valid.
    /// - The TLAS must not be in use.
    pub unsafe fn destroy(
        mut self,
        as_loader: &ash::khr::acceleration_structure::Device,
        allocator: &mut GpuAllocator,
    ) -> Result<()> {
        as_loader.destroy_acceleration_structure(self.acceleration_structure, None);
        allocator.free_buffer(&mut self.buffer)?;
        allocator.free_buffer(&mut self.instance_buffer)?;
        Ok(())
    }
}

/// Combined scene acceleration structure containing BLAS and TLAS.
///
/// This provides a convenient wrapper for managing both acceleration
/// structures and their shared scratch buffer.
pub struct SceneAccelerationStructure {
    /// Bottom-level acceleration structure (procedural AABB).
    pub blas: ProceduralBlas,
    /// Top-level acceleration structure.
    pub tlas: Tlas,
    /// Shared scratch buffer for building.
    pub scratch_buffer: GpuBuffer,
}

impl SceneAccelerationStructure {
    /// Create scene acceleration structures for an SVO-DAG.
    ///
    /// # Safety
    /// - Device and allocator must be valid.
    /// - AS loader must be properly initialized.
    pub unsafe fn new(
        device: &ash::Device,
        allocator: &mut GpuAllocator,
        as_loader: &ash::khr::acceleration_structure::Device,
        octree_depth: u32,
    ) -> Result<Self> {
        let octree_size = (1u32 << octree_depth) as f32;

        // Create BLAS
        let blas = ProceduralBlas::new(device, allocator, as_loader, octree_size)?;

        // Create TLAS
        let tlas = Tlas::new(device, allocator, as_loader, &blas)?;

        // Calculate scratch buffer size (max of BLAS and TLAS)
        let blas_sizes = blas.get_build_sizes(device, as_loader);
        let tlas_sizes = tlas.get_build_sizes(device, as_loader);
        let scratch_size = blas_sizes
            .build_scratch_size
            .max(tlas_sizes.build_scratch_size);

        // Create scratch buffer
        let scratch_buffer = allocator.create_buffer(
            scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            MemoryLocation::GpuOnly,
            "as_scratch_buffer",
        )?;

        Ok(Self {
            blas,
            tlas,
            scratch_buffer,
        })
    }

    /// Record build commands for all acceleration structures.
    ///
    /// # Safety
    /// - All handles must be valid.
    /// - The command buffer must be in recording state.
    pub unsafe fn record_build(
        &self,
        device: &ash::Device,
        as_loader: &ash::khr::acceleration_structure::Device,
        cmd: vk::CommandBuffer,
    ) {
        // Build BLAS first
        self.blas
            .record_build(device, as_loader, cmd, &self.scratch_buffer);

        // Memory barrier between BLAS and TLAS builds
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .dst_access_mask(
                vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
                    | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
            );

        let dependency_info =
            vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&barrier));

        device.cmd_pipeline_barrier2(cmd, &dependency_info);

        // Build TLAS
        self.tlas
            .record_build(device, as_loader, cmd, &self.scratch_buffer);
    }

    /// Get the TLAS for descriptor binding.
    pub fn tlas(&self) -> &Tlas {
        &self.tlas
    }

    /// Destroy all acceleration structures and free resources.
    ///
    /// # Safety
    /// - All handles must be valid.
    /// - The structures must not be in use.
    pub unsafe fn destroy(
        mut self,
        as_loader: &ash::khr::acceleration_structure::Device,
        allocator: &mut GpuAllocator,
    ) -> Result<()> {
        self.tlas.destroy(as_loader, allocator)?;
        self.blas.destroy(as_loader, allocator)?;
        allocator.free_buffer(&mut self.scratch_buffer)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_size() {
        assert_eq!(std::mem::size_of::<AabbPositions>(), 24);
    }

    #[test]
    fn aabb_from_octree() {
        let aabb = AabbPositions::from_octree_size(32.0);
        assert_eq!(aabb.min_x, 0.0);
        assert_eq!(aabb.min_y, 0.0);
        assert_eq!(aabb.min_z, 0.0);
        assert_eq!(aabb.max_x, 32.0);
        assert_eq!(aabb.max_y, 32.0);
        assert_eq!(aabb.max_z, 32.0);
    }
}
