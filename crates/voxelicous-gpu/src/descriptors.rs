//! Descriptor set management.

use crate::error::Result;
use ash::vk;

/// Descriptor set layout builder.
pub struct DescriptorSetLayoutBuilder<'a> {
    bindings: Vec<vk::DescriptorSetLayoutBinding<'a>>,
}

impl<'a> DescriptorSetLayoutBuilder<'a> {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
        }
    }

    /// Add a binding.
    pub fn binding(
        mut self,
        binding: u32,
        descriptor_type: vk::DescriptorType,
        count: u32,
        stage_flags: vk::ShaderStageFlags,
    ) -> Self {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(descriptor_type)
                .descriptor_count(count)
                .stage_flags(stage_flags),
        );
        self
    }

    /// Add a storage buffer binding.
    pub fn storage_buffer(self, binding: u32, stage_flags: vk::ShaderStageFlags) -> Self {
        self.binding(binding, vk::DescriptorType::STORAGE_BUFFER, 1, stage_flags)
    }

    /// Add a uniform buffer binding.
    pub fn uniform_buffer(self, binding: u32, stage_flags: vk::ShaderStageFlags) -> Self {
        self.binding(binding, vk::DescriptorType::UNIFORM_BUFFER, 1, stage_flags)
    }

    /// Add a storage image binding.
    pub fn storage_image(self, binding: u32, stage_flags: vk::ShaderStageFlags) -> Self {
        self.binding(binding, vk::DescriptorType::STORAGE_IMAGE, 1, stage_flags)
    }

    /// Add a sampled image binding.
    pub fn sampled_image(self, binding: u32, stage_flags: vk::ShaderStageFlags) -> Self {
        self.binding(
            binding,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            1,
            stage_flags,
        )
    }

    /// Add an acceleration structure binding (for ray tracing).
    pub fn acceleration_structure(self, binding: u32, stage_flags: vk::ShaderStageFlags) -> Self {
        self.binding(
            binding,
            vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            1,
            stage_flags,
        )
    }

    /// Build the descriptor set layout.
    ///
    /// # Safety
    /// The device must be valid.
    pub unsafe fn build(self, device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&self.bindings);

        let layout = device.create_descriptor_set_layout(&layout_info, None)?;
        Ok(layout)
    }
}

impl Default for DescriptorSetLayoutBuilder<'_> {
    fn default() -> Self {
        Self::new()
    }
}

/// Descriptor pool for allocating descriptor sets.
pub struct DescriptorPool {
    pool: vk::DescriptorPool,
}

impl DescriptorPool {
    /// Create a new descriptor pool.
    ///
    /// # Safety
    /// The device must be valid.
    pub unsafe fn new(
        device: &ash::Device,
        max_sets: u32,
        pool_sizes: &[vk::DescriptorPoolSize],
    ) -> Result<Self> {
        let create_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        let pool = device.create_descriptor_pool(&create_info, None)?;
        Ok(Self { pool })
    }

    /// Get the raw pool handle.
    pub fn handle(&self) -> vk::DescriptorPool {
        self.pool
    }

    /// Allocate descriptor sets.
    ///
    /// # Safety
    /// The device must be valid.
    pub unsafe fn allocate(
        &self,
        device: &ash::Device,
        layouts: &[vk::DescriptorSetLayout],
    ) -> Result<Vec<vk::DescriptorSet>> {
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(layouts);

        let sets = device.allocate_descriptor_sets(&alloc_info)?;
        Ok(sets)
    }

    /// Reset the pool, freeing all descriptor sets.
    ///
    /// # Safety
    /// The device must be valid and no descriptor sets must be in use.
    pub unsafe fn reset(&self, device: &ash::Device) -> Result<()> {
        device.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())?;
        Ok(())
    }

    /// Destroy the pool.
    ///
    /// # Safety
    /// The device must be valid and the pool must not be in use.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_descriptor_pool(self.pool, None);
    }
}

/// Write a uniform buffer descriptor.
///
/// # Safety
/// Device and buffer must be valid.
pub unsafe fn write_uniform_buffer(
    device: &ash::Device,
    dst_set: vk::DescriptorSet,
    binding: u32,
    buffer: vk::Buffer,
    offset: u64,
    range: u64,
) {
    let buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .offset(offset)
        .range(range);

    let write = vk::WriteDescriptorSet::default()
        .dst_set(dst_set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(std::slice::from_ref(&buffer_info));

    device.update_descriptor_sets(&[write], &[]);
}

/// Write a storage buffer descriptor.
///
/// # Safety
/// Device and buffer must be valid.
pub unsafe fn write_storage_buffer(
    device: &ash::Device,
    dst_set: vk::DescriptorSet,
    binding: u32,
    buffer: vk::Buffer,
    offset: u64,
    range: u64,
) {
    let buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .offset(offset)
        .range(range);

    let write = vk::WriteDescriptorSet::default()
        .dst_set(dst_set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(std::slice::from_ref(&buffer_info));

    device.update_descriptor_sets(&[write], &[]);
}

/// Write a storage image descriptor.
///
/// # Safety
/// Device and image view must be valid.
pub unsafe fn write_storage_image(
    device: &ash::Device,
    dst_set: vk::DescriptorSet,
    binding: u32,
    image_view: vk::ImageView,
    layout: vk::ImageLayout,
) {
    let image_info = vk::DescriptorImageInfo::default()
        .image_view(image_view)
        .image_layout(layout);

    let write = vk::WriteDescriptorSet::default()
        .dst_set(dst_set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .image_info(std::slice::from_ref(&image_info));

    device.update_descriptor_sets(&[write], &[]);
}
