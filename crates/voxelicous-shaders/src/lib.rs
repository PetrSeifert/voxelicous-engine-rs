//! Shader compilation for the Voxelicous engine.
//!
//! This crate contains GLSL shaders and their compiled SPIR-V bytecode.
//! Shaders are compiled at build time using shaderc.

use std::sync::OnceLock;

/// Embedded SPIR-V shader bytecode (raw bytes, may not be aligned).
mod spirv_bytes {
    /// Ray march world compute shader for multi-chunk rendering (compiled SPIR-V).
    pub static RAY_MARCH_WORLD_COMP: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/ray_march_world.spv"));
}

/// Convert byte slice to aligned u32 Vec (SPIR-V requires 4-byte alignment).
fn bytes_to_spirv(bytes: &[u8]) -> Vec<u32> {
    assert!(
        bytes.len() % 4 == 0,
        "SPIR-V bytecode must be 4-byte aligned"
    );
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Cached aligned SPIR-V for world ray march shader.
static RAY_MARCH_WORLD_SPIRV: OnceLock<Vec<u32>> = OnceLock::new();

/// Get ray march world shader for multi-chunk rendering as u32 slice for Vulkan.
pub fn ray_march_world_shader() -> &'static [u32] {
    RAY_MARCH_WORLD_SPIRV.get_or_init(|| bytes_to_spirv(spirv_bytes::RAY_MARCH_WORLD_COMP))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_shader_loads() {
        let shader = ray_march_world_shader();
        // SPIR-V magic number is 0x07230203
        assert_eq!(shader[0], 0x0723_0203, "Invalid SPIR-V magic number");
        assert!(shader.len() > 100, "Shader too small");
    }
}
