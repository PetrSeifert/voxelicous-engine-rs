//! GPU memory management.

use crate::error::{GpuError, Result};
use ash::vk;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::MemoryLocation;
use std::sync::Arc;

/// GPU memory allocator wrapper.
pub struct GpuAllocator {
    allocator: Option<Allocator>,
    device: Arc<ash::Device>,
}

impl GpuAllocator {
    /// Create a new allocator.
    ///
    /// # Safety
    /// The instance, device, and physical device must be valid.
    pub unsafe fn new(
        instance: &ash::Instance,
        device: Arc<ash::Device>,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: (*device).clone(),
            physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings {
                log_memory_information: cfg!(debug_assertions),
                log_leaks_on_shutdown: true,
                store_stack_traces: cfg!(debug_assertions),
                log_allocations: false,
                log_frees: false,
                log_stack_traces: false,
            },
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })
        .map_err(|e| GpuError::AllocationFailed(e.to_string()))?;

        Ok(Self {
            allocator: Some(allocator),
            device,
        })
    }

    /// Allocate a buffer.
    pub fn create_buffer(
        &mut self,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<GpuBuffer> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device
                .create_buffer(&buffer_info, None)
                .map_err(GpuError::from)?
        };

        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocation = self
            .allocator
            .as_mut()
            .ok_or_else(|| GpuError::InvalidState("Allocator not initialized".to_string()))?
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| GpuError::AllocationFailed(e.to_string()))?;

        unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(GpuError::from)?;
        }

        Ok(GpuBuffer {
            buffer,
            allocation: Some(allocation),
            size,
        })
    }

    /// Free a buffer allocation.
    pub fn free_buffer(&mut self, buffer: &mut GpuBuffer) -> Result<()> {
        if let Some(allocation) = buffer.allocation.take() {
            self.allocator
                .as_mut()
                .ok_or_else(|| GpuError::InvalidState("Allocator not initialized".to_string()))?
                .free(allocation)
                .map_err(|e| GpuError::AllocationFailed(e.to_string()))?;
        }

        unsafe {
            self.device.destroy_buffer(buffer.buffer, None);
        }
        buffer.buffer = vk::Buffer::null();

        Ok(())
    }

    /// Allocate an image.
    pub fn create_image(
        &mut self,
        create_info: &vk::ImageCreateInfo,
        location: MemoryLocation,
        name: &str,
    ) -> Result<GpuImage> {
        let image = unsafe {
            self.device
                .create_image(create_info, None)
                .map_err(GpuError::from)?
        };

        let requirements = unsafe { self.device.get_image_memory_requirements(image) };

        let allocation = self
            .allocator
            .as_mut()
            .ok_or_else(|| GpuError::InvalidState("Allocator not initialized".to_string()))?
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| GpuError::AllocationFailed(e.to_string()))?;

        unsafe {
            self.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .map_err(GpuError::from)?;
        }

        Ok(GpuImage {
            image,
            allocation: Some(allocation),
            format: create_info.format,
            extent: create_info.extent,
        })
    }

    /// Free an image allocation.
    pub fn free_image(&mut self, image: &mut GpuImage) -> Result<()> {
        if let Some(allocation) = image.allocation.take() {
            self.allocator
                .as_mut()
                .ok_or_else(|| GpuError::InvalidState("Allocator not initialized".to_string()))?
                .free(allocation)
                .map_err(|e| GpuError::AllocationFailed(e.to_string()))?;
        }

        unsafe {
            self.device.destroy_image(image.image, None);
        }
        image.image = vk::Image::null();

        Ok(())
    }
}

impl GpuAllocator {
    /// Shutdown the allocator, freeing all GPU memory.
    ///
    /// This must be called before the Vulkan device is destroyed.
    /// Any remaining allocations will be freed (and logged as leaks).
    pub fn shutdown(&mut self) {
        // Take and drop the inner allocator to free all GPU memory
        // The gpu_allocator::Allocator::Drop will call vkFreeMemory
        if let Some(allocator) = self.allocator.take() {
            drop(allocator);
        }
    }
}

impl Drop for GpuAllocator {
    fn drop(&mut self) {
        // Shutdown if not already done
        self.shutdown();
    }
}

/// A GPU buffer with its allocation.
pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size: u64,
}

impl GpuBuffer {
    /// Get the device address of this buffer.
    pub fn device_address(&self, device: &ash::Device) -> vk::DeviceAddress {
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        unsafe { device.get_buffer_device_address(&info) }
    }

    /// Map the buffer memory for CPU access.
    pub fn mapped_ptr(&self) -> Option<*mut u8> {
        self.allocation
            .as_ref()
            .and_then(|a| a.mapped_ptr())
            .map(|p| p.as_ptr() as *mut u8)
    }

    /// Write data to the buffer (must be host-visible).
    pub fn write<T: Copy>(&self, data: &[T]) -> Result<()> {
        let ptr = self
            .mapped_ptr()
            .ok_or_else(|| GpuError::InvalidState("Buffer not mapped".to_string()))?;

        let byte_size = std::mem::size_of_val(data);
        if byte_size as u64 > self.size {
            return Err(GpuError::InvalidState(
                "Data too large for buffer".to_string(),
            ));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr, byte_size);
        }

        Ok(())
    }

    /// Write raw bytes to the buffer at the given offset (must be host-visible).
    pub fn write_bytes(&self, offset: u64, data: &[u8]) -> Result<()> {
        let ptr = self
            .mapped_ptr()
            .ok_or_else(|| GpuError::InvalidState("Buffer not mapped".to_string()))?;

        let end = offset
            .checked_add(data.len() as u64)
            .ok_or_else(|| GpuError::InvalidState("Offset overflow".to_string()))?;
        if end > self.size {
            return Err(GpuError::InvalidState(
                "Data range too large for buffer".to_string(),
            ));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset as usize), data.len());
        }

        Ok(())
    }

    /// Write typed data to the buffer at the given offset (must be host-visible).
    pub fn write_range<T: Copy>(&self, offset: u64, data: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        self.write_bytes(offset, unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes)
        })
    }
}

/// A GPU image with its allocation.
pub struct GpuImage {
    pub image: vk::Image,
    pub allocation: Option<Allocation>,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
}
