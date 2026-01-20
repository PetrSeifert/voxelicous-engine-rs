//! Compute shader ray marching implementation.
//!
//! This is the software fallback for GPUs without hardware ray tracing.

use glam::Vec3;

/// Ray for marching through voxel data.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }

    /// Get point along ray at distance t.
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

/// Hit result from ray marching.
#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    pub t: f32,
    pub position: Vec3,
    pub normal: Vec3,
    pub block_id: u32,
}

/// Configuration for the ray marcher.
#[derive(Debug, Clone)]
pub struct RayMarchConfig {
    pub max_steps: u32,
    pub max_distance: f32,
    pub epsilon: f32,
}

impl Default for RayMarchConfig {
    fn default() -> Self {
        Self {
            max_steps: 256,
            max_distance: 1000.0,
            epsilon: 0.001,
        }
    }
}

/// Push constants for ray marching compute shader.
///
/// This structure must match the layout in `ray_march_svo.comp`:
/// ```glsl
/// layout(push_constant) uniform PushConstants {
///     uvec2 screen_size;      // 8 bytes
///     uint max_steps;         // 4 bytes
///     uint _padding;          // 4 bytes
///     uint64_t node_buffer_address;  // 8 bytes
///     uint root_index;        // 4 bytes
///     uint octree_depth;      // 4 bytes
/// } pc;
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RayMarchPushConstants {
    /// Render target dimensions (width, height).
    pub screen_size: [u32; 2],
    /// Maximum traversal steps.
    pub max_steps: u32,
    /// Padding for alignment.
    pub _padding: u32,
    /// Device address of the SVO-DAG node buffer.
    pub node_buffer_address: u64,
    /// Root node index (0 = empty octree).
    pub root_index: u32,
    /// Octree depth (determines size as 2^depth).
    pub octree_depth: u32,
}

impl RayMarchPushConstants {
    /// Size in bytes (must be 32 bytes for shader alignment).
    pub const SIZE: u32 = std::mem::size_of::<Self>() as u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_constants_size() {
        // Ensure push constants are exactly 32 bytes
        assert_eq!(RayMarchPushConstants::SIZE, 32);
    }

    #[test]
    fn push_constants_layout() {
        // Verify offsets match shader expectations
        assert_eq!(std::mem::offset_of!(RayMarchPushConstants, screen_size), 0);
        assert_eq!(std::mem::offset_of!(RayMarchPushConstants, max_steps), 8);
        assert_eq!(std::mem::offset_of!(RayMarchPushConstants, _padding), 12);
        assert_eq!(
            std::mem::offset_of!(RayMarchPushConstants, node_buffer_address),
            16
        );
        assert_eq!(std::mem::offset_of!(RayMarchPushConstants, root_index), 24);
        assert_eq!(
            std::mem::offset_of!(RayMarchPushConstants, octree_depth),
            28
        );
    }
}
