//! Entity system for the Voxelicous engine.
//!
//! Uses hecs as the ECS backend.

use glam::Vec3;
pub use hecs::{Entity, World};

/// Transform component.
#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: glam::Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}
