//! Pipeline creation and management.

use crate::error::{GpuError, Result};
use ash::vk;

/// Compute pipeline wrapper.
pub struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

impl ComputePipeline {
    /// Create a compute pipeline from shader code.
    ///
    /// # Safety
    /// The device must be valid and the shader code must be valid SPIR-V.
    pub unsafe fn new(
        device: &ash::Device,
        shader_code: &[u32],
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self> {
        // Create shader module
        let shader_info = vk::ShaderModuleCreateInfo::default().code(shader_code);
        let shader_module = device
            .create_shader_module(&shader_info, None)
            .map_err(|e| GpuError::ShaderCompilation(e.to_string()))?;

        // Create pipeline layout
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);

        let layout = device
            .create_pipeline_layout(&layout_info, None)
            .map_err(|e| GpuError::PipelineCreation(e.to_string()))?;

        // Create compute pipeline
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(layout);

        let pipelines = device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_pipelines, e)| GpuError::PipelineCreation(e.to_string()))?;

        // Clean up shader module (no longer needed)
        device.destroy_shader_module(shader_module, None);

        Ok(Self {
            pipeline: pipelines[0],
            layout,
        })
    }

    /// Destroy the pipeline.
    ///
    /// # Safety
    /// The device must be valid and the pipeline must not be in use.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.layout, None);
    }
}

/// Graphics pipeline configuration.
#[derive(Clone)]
pub struct GraphicsPipelineConfig {
    pub vertex_shader: Vec<u32>,
    pub fragment_shader: Vec<u32>,
    pub vertex_bindings: Vec<vk::VertexInputBindingDescription>,
    pub vertex_attributes: Vec<vk::VertexInputAttributeDescription>,
    pub topology: vk::PrimitiveTopology,
    pub polygon_mode: vk::PolygonMode,
    pub cull_mode: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub depth_test: bool,
    pub depth_write: bool,
    pub color_formats: Vec<vk::Format>,
    pub depth_format: Option<vk::Format>,
}

impl Default for GraphicsPipelineConfig {
    fn default() -> Self {
        Self {
            vertex_shader: Vec::new(),
            fragment_shader: Vec::new(),
            vertex_bindings: Vec::new(),
            vertex_attributes: Vec::new(),
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_test: true,
            depth_write: true,
            color_formats: vec![vk::Format::B8G8R8A8_SRGB],
            depth_format: Some(vk::Format::D32_SFLOAT),
        }
    }
}

/// Graphics pipeline wrapper.
pub struct GraphicsPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

impl GraphicsPipeline {
    /// Create a graphics pipeline using dynamic rendering (Vulkan 1.3).
    ///
    /// # Safety
    /// The device must be valid and shader code must be valid SPIR-V.
    pub unsafe fn new(
        device: &ash::Device,
        config: &GraphicsPipelineConfig,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self> {
        // Create shader modules
        let vert_shader_info = vk::ShaderModuleCreateInfo::default().code(&config.vertex_shader);
        let vert_module = device
            .create_shader_module(&vert_shader_info, None)
            .map_err(|e| GpuError::ShaderCompilation(format!("Vertex: {e}")))?;

        let frag_shader_info = vk::ShaderModuleCreateInfo::default().code(&config.fragment_shader);
        let frag_module = device
            .create_shader_module(&frag_shader_info, None)
            .map_err(|e| GpuError::ShaderCompilation(format!("Fragment: {e}")))?;

        // Shader stages
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(c"main"),
        ];

        // Vertex input
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&config.vertex_bindings)
            .vertex_attribute_descriptions(&config.vertex_attributes);

        // Input assembly
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(config.topology)
            .primitive_restart_enable(false);

        // Viewport (dynamic)
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        // Rasterization
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(config.polygon_mode)
            .cull_mode(config.cull_mode)
            .front_face(config.front_face)
            .depth_bias_enable(false)
            .line_width(1.0);

        // Multisampling
        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false);

        // Depth stencil
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(config.depth_test)
            .depth_write_enable(config.depth_write)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        // Color blending
        let color_blend_attachments: Vec<_> = config
            .color_formats
            .iter()
            .map(|_| {
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(false)
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
            })
            .collect();

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&color_blend_attachments);

        // Dynamic state
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // Pipeline layout
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);

        let layout = device
            .create_pipeline_layout(&layout_info, None)
            .map_err(|e| GpuError::PipelineCreation(e.to_string()))?;

        // Dynamic rendering info (Vulkan 1.3)
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&config.color_formats);

        if let Some(depth_format) = config.depth_format {
            rendering_info = rendering_info.depth_attachment_format(depth_format);
        }

        // Create pipeline
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .push_next(&mut rendering_info);

        let pipelines = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(_pipelines, e)| GpuError::PipelineCreation(e.to_string()))?;

        // Clean up shader modules
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);

        Ok(Self {
            pipeline: pipelines[0],
            layout,
        })
    }

    /// Destroy the pipeline.
    ///
    /// # Safety
    /// The device must be valid and the pipeline must not be in use.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.layout, None);
    }
}
