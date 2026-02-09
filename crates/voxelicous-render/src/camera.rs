//! Camera and view management.

use glam::{Mat4, Vec3};
use voxelicous_core::math::Frustum;

/// Camera for rendering.
#[derive(Debug, Clone)]
pub struct Camera {
    pub position: Vec3,
    pub direction: Vec3,
    pub up: Vec3,
    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            direction: Vec3::NEG_Z,
            up: Vec3::Y,
            fov: std::f32::consts::FRAC_PI_4,
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 1000.0,
        }
    }
}

impl Camera {
    /// Create a new camera.
    pub fn new(
        position: Vec3,
        target: Vec3,
        up: Vec3,
        fov: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let direction = (target - position).normalize();
        Self {
            position,
            direction,
            up,
            fov,
            aspect,
            near,
            far,
        }
    }

    /// Set the camera position.
    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }

    /// Look at a target position.
    pub fn look_at(&mut self, target: Vec3) {
        self.direction = (target - self.position).normalize();
    }

    /// Set the aspect ratio.
    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }

    /// Get camera uniforms for GPU.
    pub fn uniforms(&self) -> CameraUniforms {
        self.uniforms_with_day_phase(0.25)
    }

    /// Get camera uniforms for GPU with a normalized day phase.
    ///
    /// `day_phase` wraps to the range `[0.0, 1.0)`, where:
    /// - `0.25` is noon
    /// - `0.75` is midnight
    pub fn uniforms_with_day_phase(&self, day_phase: f32) -> CameraUniforms {
        let mut uniforms = CameraUniforms::from(self);
        uniforms.day_night = [day_phase.rem_euclid(1.0), 0.0, 0.0, 0.0];
        uniforms
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.direction, self.up)
    }

    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far)
    }

    pub fn inverse_view_matrix(&self) -> Mat4 {
        self.view_matrix().inverse()
    }

    pub fn inverse_projection_matrix(&self) -> Mat4 {
        self.projection_matrix().inverse()
    }

    /// Get the view-projection matrix.
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// Extract frustum planes from the current camera state.
    pub fn frustum(&self) -> Frustum {
        Frustum::from_view_projection(self.view_projection_matrix())
    }
}

/// Camera uniform buffer data for GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    pub view: [[f32; 4]; 4],
    pub projection: [[f32; 4]; 4],
    pub inverse_view: [[f32; 4]; 4],
    pub inverse_projection: [[f32; 4]; 4],
    pub position: [f32; 4],
    pub direction: [f32; 4],
    pub day_night: [f32; 4],
}

impl From<&Camera> for CameraUniforms {
    fn from(camera: &Camera) -> Self {
        Self {
            view: camera.view_matrix().to_cols_array_2d(),
            projection: camera.projection_matrix().to_cols_array_2d(),
            inverse_view: camera.inverse_view_matrix().to_cols_array_2d(),
            inverse_projection: camera.inverse_projection_matrix().to_cols_array_2d(),
            position: [camera.position.x, camera.position.y, camera.position.z, 1.0],
            direction: [
                camera.direction.x,
                camera.direction.y,
                camera.direction.z,
                0.0,
            ],
            day_night: [0.25, 0.0, 0.0, 0.0],
        }
    }
}
