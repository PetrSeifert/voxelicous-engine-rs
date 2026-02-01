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

    /// Ray tracing shaders (feature-gated).
    #[cfg(feature = "ray_tracing")]
    pub mod ray_tracing {
        /// Ray generation shader for SVO-DAG ray tracing.
        pub static RAY_TRACE_SVO_RGEN: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/ray_trace_svo_rgen.spv"));

        /// Miss shader for SVO-DAG ray tracing (sky rendering).
        pub static RAY_TRACE_SVO_RMISS: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/ray_trace_svo_rmiss.spv"));

        /// Intersection shader for SVO-DAG traversal.
        pub static RAY_TRACE_SVO_RINT: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/ray_trace_svo_rint.spv"));

        /// Closest-hit shader for SVO-DAG voxel shading.
        pub static RAY_TRACE_SVO_RCHIT: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/ray_trace_svo_rchit.spv"));
    }
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

/// Cached aligned SPIR-V for ray tracing shaders.
#[cfg(feature = "ray_tracing")]
mod ray_tracing_cache {
    use super::*;
    pub static RAYGEN: OnceLock<Vec<u32>> = OnceLock::new();
    pub static MISS: OnceLock<Vec<u32>> = OnceLock::new();
    pub static INTERSECTION: OnceLock<Vec<u32>> = OnceLock::new();
    pub static CLOSEST_HIT: OnceLock<Vec<u32>> = OnceLock::new();
}

/// Get ray generation shader for hardware ray tracing.
#[cfg(feature = "ray_tracing")]
pub fn ray_trace_svo_raygen_shader() -> &'static [u32] {
    ray_tracing_cache::RAYGEN
        .get_or_init(|| bytes_to_spirv(spirv_bytes::ray_tracing::RAY_TRACE_SVO_RGEN))
}

/// Get miss shader for hardware ray tracing.
#[cfg(feature = "ray_tracing")]
pub fn ray_trace_svo_miss_shader() -> &'static [u32] {
    ray_tracing_cache::MISS
        .get_or_init(|| bytes_to_spirv(spirv_bytes::ray_tracing::RAY_TRACE_SVO_RMISS))
}

/// Get intersection shader for hardware ray tracing.
#[cfg(feature = "ray_tracing")]
pub fn ray_trace_svo_intersection_shader() -> &'static [u32] {
    ray_tracing_cache::INTERSECTION
        .get_or_init(|| bytes_to_spirv(spirv_bytes::ray_tracing::RAY_TRACE_SVO_RINT))
}

/// Get closest-hit shader for hardware ray tracing.
#[cfg(feature = "ray_tracing")]
pub fn ray_trace_svo_closest_hit_shader() -> &'static [u32] {
    ray_tracing_cache::CLOSEST_HIT
        .get_or_init(|| bytes_to_spirv(spirv_bytes::ray_tracing::RAY_TRACE_SVO_RCHIT))
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

    #[cfg(feature = "ray_tracing")]
    #[test]
    fn ray_tracing_shaders_load() {
        let shaders = [
            ("raygen", ray_trace_svo_raygen_shader()),
            ("miss", ray_trace_svo_miss_shader()),
            ("intersection", ray_trace_svo_intersection_shader()),
            ("closest_hit", ray_trace_svo_closest_hit_shader()),
        ];

        for (name, shader) in shaders {
            assert_eq!(
                shader[0], 0x0723_0203,
                "Invalid SPIR-V magic number for {name} shader"
            );
            assert!(shader.len() > 10, "{name} shader too small");
        }
    }
}
