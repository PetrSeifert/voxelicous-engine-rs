//! Math utilities and helpers.

use glam::{Mat4, Vec3, Vec4};

/// Ray for raycasting operations.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    /// Ray origin
    pub origin: Vec3,
    /// Ray direction (should be normalized)
    pub direction: Vec3,
}

impl Ray {
    /// Create a new ray
    #[inline]
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }

    /// Get a point along the ray at distance t
    #[inline]
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// Transform ray by a matrix
    #[inline]
    pub fn transform(&self, matrix: Mat4) -> Self {
        let origin = matrix.transform_point3(self.origin);
        let direction = matrix.transform_vector3(self.direction).normalize();
        Self { origin, direction }
    }
}

/// Axis-Aligned Bounding Box.
#[derive(Clone, Copy, Debug, Default)]
pub struct Aabb {
    /// Minimum corner
    pub min: Vec3,
    /// Maximum corner
    pub max: Vec3,
}

impl Aabb {
    /// Create a new AABB from min and max corners
    #[inline]
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Create an AABB centered at origin with given half-extents
    #[inline]
    pub fn from_half_extents(half_extents: Vec3) -> Self {
        Self {
            min: -half_extents,
            max: half_extents,
        }
    }

    /// Create an AABB for a unit cube at the given position
    #[inline]
    pub fn unit_cube(pos: Vec3) -> Self {
        Self {
            min: pos,
            max: pos + Vec3::ONE,
        }
    }

    /// Get the center of the AABB
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Get the size of the AABB
    #[inline]
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// Get the half-extents of the AABB
    #[inline]
    pub fn half_extents(&self) -> Vec3 {
        self.size() * 0.5
    }

    /// Check if a point is inside the AABB
    #[inline]
    pub fn contains_point(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Check if this AABB intersects another
    #[inline]
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Ray-AABB intersection test, returns (t_near, t_far) or None if no intersection
    pub fn intersect_ray(&self, ray: &Ray) -> Option<(f32, f32)> {
        let inv_dir = Vec3::ONE / ray.direction;

        let t1 = (self.min - ray.origin) * inv_dir;
        let t2 = (self.max - ray.origin) * inv_dir;

        let t_min = t1.min(t2);
        let t_max = t1.max(t2);

        let t_near = t_min.x.max(t_min.y).max(t_min.z);
        let t_far = t_max.x.min(t_max.y).min(t_max.z);

        if t_near <= t_far && t_far >= 0.0 {
            Some((t_near.max(0.0), t_far))
        } else {
            None
        }
    }

    /// Expand AABB to include a point
    #[inline]
    pub fn expand_to_include(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    /// Merge two AABBs
    #[inline]
    pub fn merge(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

/// Frustum for culling operations.
#[derive(Clone, Copy, Debug)]
pub struct Frustum {
    /// Six frustum planes (left, right, bottom, top, near, far)
    /// Each plane is (nx, ny, nz, d) where n is normal and d is distance
    pub planes: [Vec4; 6],
}

impl Frustum {
    /// Extract frustum planes from view-projection matrix
    pub fn from_view_projection(vp: Mat4) -> Self {
        let row0 = vp.row(0);
        let row1 = vp.row(1);
        let row2 = vp.row(2);
        let row3 = vp.row(3);

        let planes = [
            (row3 + row0).normalize(), // Left
            (row3 - row0).normalize(), // Right
            (row3 + row1).normalize(), // Bottom
            (row3 - row1).normalize(), // Top
            (row3 + row2).normalize(), // Near
            (row3 - row2).normalize(), // Far
        ];

        Self { planes }
    }

    /// Test if an AABB is inside or intersects the frustum
    pub fn test_aabb(&self, aabb: &Aabb) -> bool {
        for plane in &self.planes {
            let normal = Vec3::new(plane.x, plane.y, plane.z);

            // Find the positive vertex (furthest along plane normal)
            let p = Vec3::new(
                if normal.x >= 0.0 {
                    aabb.max.x
                } else {
                    aabb.min.x
                },
                if normal.y >= 0.0 {
                    aabb.max.y
                } else {
                    aabb.min.y
                },
                if normal.z >= 0.0 {
                    aabb.max.z
                } else {
                    aabb.min.z
                },
            );

            if normal.dot(p) + plane.w < 0.0 {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_at() {
        let ray = Ray::new(Vec3::ZERO, Vec3::X);
        assert_eq!(ray.at(0.0), Vec3::ZERO);
        assert_eq!(ray.at(1.0), Vec3::X);
        assert_eq!(ray.at(5.0), Vec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn aabb_contains_point() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);
        assert!(aabb.contains_point(Vec3::splat(0.5)));
        assert!(aabb.contains_point(Vec3::ZERO));
        assert!(aabb.contains_point(Vec3::ONE));
        assert!(!aabb.contains_point(Vec3::new(2.0, 0.5, 0.5)));
    }

    #[test]
    fn aabb_ray_intersection() {
        let aabb = Aabb::new(Vec3::ZERO, Vec3::ONE);

        // Ray hitting the box
        let ray = Ray::new(Vec3::new(-1.0, 0.5, 0.5), Vec3::X);
        let hit = aabb.intersect_ray(&ray);
        assert!(hit.is_some());
        let (t_near, t_far) = hit.unwrap();
        assert!((t_near - 1.0).abs() < 0.001);
        assert!((t_far - 2.0).abs() < 0.001);

        // Ray missing the box
        let ray = Ray::new(Vec3::new(-1.0, 2.0, 0.5), Vec3::X);
        assert!(aabb.intersect_ray(&ray).is_none());
    }
}
