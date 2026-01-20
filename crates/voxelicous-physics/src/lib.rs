//! Voxel collision and physics for the Voxelicous engine.

use glam::Vec3;

/// Ray for collision detection.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

/// Result of a raycast against voxels.
#[derive(Debug, Clone, Copy)]
pub struct RaycastHit {
    pub position: Vec3,
    pub normal: Vec3,
    pub distance: f32,
    pub block_position: [i32; 3],
}
