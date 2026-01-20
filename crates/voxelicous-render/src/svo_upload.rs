//! GPU upload mechanism for SVO-DAG data.
//!
//! Handles transferring `SvoDag` data to GPU buffers for shader access.

use ash::vk;
use gpu_allocator::MemoryLocation;
use voxelicous_gpu::error::Result;
use voxelicous_gpu::memory::{GpuAllocator, GpuBuffer};
use voxelicous_voxel::{GpuOctreeNode, SvoDag, VoxelStorage};

/// GPU-resident SVO-DAG data ready for shader access.
///
/// Contains the node buffer and metadata needed for ray traversal.
pub struct GpuSvoDag {
    /// Storage buffer containing `GpuOctreeNode` array.
    pub node_buffer: GpuBuffer,
    /// Total number of nodes in the buffer.
    pub node_count: u32,
    /// Root node index (0 = empty octree).
    pub root_index: u32,
    /// Octree depth (determines octree size as 2^depth).
    pub depth: u32,
    /// Device address for buffer reference in shaders.
    pub device_address: vk::DeviceAddress,
}

impl GpuSvoDag {
    /// Upload an SvoDag to GPU memory.
    ///
    /// Creates a storage buffer with shader device address support
    /// for access via buffer references in compute/ray tracing shaders.
    ///
    /// # Arguments
    /// * `allocator` - GPU memory allocator
    /// * `device` - Vulkan device handle
    /// * `dag` - The SVO-DAG to upload
    ///
    /// # Returns
    /// A `GpuSvoDag` containing the uploaded buffer and metadata.
    pub fn upload(
        allocator: &mut GpuAllocator,
        device: &ash::Device,
        dag: &SvoDag,
    ) -> Result<Self> {
        // Convert DAG to GPU buffer format
        let nodes = dag.to_gpu_buffer();
        let node_count = nodes.len() as u32;

        // Calculate buffer size
        let buffer_size = (nodes.len() * std::mem::size_of::<GpuOctreeNode>()) as u64;

        // Create storage buffer with device address support
        let usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        let node_buffer = allocator.create_buffer(
            buffer_size,
            usage,
            MemoryLocation::CpuToGpu, // Host-visible for direct upload
            "svo_dag_nodes",
        )?;

        // Write node data to buffer
        node_buffer.write::<u8>(bytemuck::cast_slice(&nodes))?;

        // Get device address for shader access
        let device_address = node_buffer.device_address(device);

        Ok(Self {
            node_buffer,
            node_count,
            root_index: dag.root(),
            depth: dag.depth(),
            device_address,
        })
    }

    /// Upload an SvoDag using a staging buffer (for GPU-only memory).
    ///
    /// This method uses a staging buffer to upload data to GPU-only memory,
    /// which may provide better performance for large datasets.
    ///
    /// # Arguments
    /// * `allocator` - GPU memory allocator
    /// * `device` - Vulkan device handle
    /// * `dag` - The SVO-DAG to upload
    /// * `command_buffer` - Command buffer for recording copy commands
    /// * `queue` - Queue for submission
    ///
    /// # Returns
    /// A `GpuSvoDag` containing the uploaded buffer and metadata.
    ///
    /// # Safety
    /// The command buffer must be in recording state.
    pub unsafe fn upload_staged(
        allocator: &mut GpuAllocator,
        device: &ash::Device,
        dag: &SvoDag,
        command_buffer: vk::CommandBuffer,
        queue: vk::Queue,
        fence: vk::Fence,
    ) -> Result<Self> {
        let nodes = dag.to_gpu_buffer();
        let node_count = nodes.len() as u32;
        let buffer_size = (nodes.len() * std::mem::size_of::<GpuOctreeNode>()) as u64;

        // Create staging buffer (CPU-visible)
        let mut staging_buffer = allocator.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "svo_dag_staging",
        )?;

        // Write to staging buffer
        staging_buffer.write::<u8>(bytemuck::cast_slice(&nodes))?;

        // Create GPU buffer
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_DST;

        let node_buffer = allocator.create_buffer(
            buffer_size,
            usage,
            MemoryLocation::GpuOnly,
            "svo_dag_nodes",
        )?;

        // Record copy command
        let copy_region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: buffer_size,
        };

        device.cmd_copy_buffer(
            command_buffer,
            staging_buffer.buffer,
            node_buffer.buffer,
            &[copy_region],
        );

        // End command buffer
        device.end_command_buffer(command_buffer)?;

        // Submit and wait
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

        device.queue_submit(queue, &[submit_info], fence)?;
        device.wait_for_fences(&[fence], true, u64::MAX)?;
        device.reset_fences(&[fence])?;

        // Free staging buffer
        allocator.free_buffer(&mut staging_buffer)?;

        let device_address = node_buffer.device_address(device);

        Ok(Self {
            node_buffer,
            node_count,
            root_index: dag.root(),
            depth: dag.depth(),
            device_address,
        })
    }

    /// Check if the SVO-DAG is empty.
    pub fn is_empty(&self) -> bool {
        self.root_index == 0
    }

    /// Get the size of the octree (2^depth voxels per axis).
    pub fn octree_size(&self) -> u32 {
        1 << self.depth
    }

    /// Get the GPU buffer size in bytes.
    pub fn buffer_size(&self) -> u64 {
        self.node_buffer.size
    }

    /// Free GPU resources.
    ///
    /// # Arguments
    /// * `allocator` - GPU memory allocator used to create the buffer
    pub fn destroy(mut self, allocator: &mut GpuAllocator) -> Result<()> {
        allocator.free_buffer(&mut self.node_buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_svo_dag_layout() {
        // Verify GpuOctreeNode is 48 bytes (matching shader expectation)
        assert_eq!(std::mem::size_of::<GpuOctreeNode>(), 48);
    }
}
