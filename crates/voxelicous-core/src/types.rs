//! Core voxel types.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// Unique identifier for a block type.
///
/// Block ID 0 is reserved for air (empty space).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable, Serialize, Deserialize,
)]
#[repr(transparent)]
pub struct BlockId(pub u16);

impl BlockId {
    /// Air block (empty space)
    pub const AIR: Self = Self(0);
    /// Stone block
    pub const STONE: Self = Self(1);
    /// Dirt block
    pub const DIRT: Self = Self(2);
    /// Grass block
    pub const GRASS: Self = Self(3);
    /// Snow block
    pub const SNOW: Self = Self(4);
    /// Sand block
    pub const SAND: Self = Self(5);
    /// Water block (opaque for current renderer path)
    pub const WATER: Self = Self(6);
    /// Tree log block
    pub const LOG: Self = Self(7);
    /// Tree leaves block
    pub const LEAVES: Self = Self(8);
    /// Flower block
    pub const FLOWER: Self = Self(9);

    /// Returns true if this block is air (empty)
    #[inline]
    pub const fn is_air(self) -> bool {
        self.0 == 0
    }

    /// Returns true if this block is solid (not air)
    #[inline]
    pub const fn is_solid(self) -> bool {
        self.0 != 0
    }
}

/// Material properties for rendering.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Material {
    /// Base color (RGB, 0-255)
    pub color: [u8; 3],
    /// Roughness (0.0 = mirror, 1.0 = diffuse)
    pub roughness: f32,
    /// Metallic (0.0 = dielectric, 1.0 = metal)
    pub metallic: f32,
    /// Emission strength (0.0 = no emission)
    pub emission: f32,
}

impl Material {
    /// Default stone material
    pub const STONE: Self = Self {
        color: [128, 128, 128],
        roughness: 0.8,
        metallic: 0.0,
        emission: 0.0,
    };

    /// Default dirt material
    pub const DIRT: Self = Self {
        color: [139, 90, 43],
        roughness: 0.9,
        metallic: 0.0,
        emission: 0.0,
    };

    /// Default grass material
    pub const GRASS: Self = Self {
        color: [86, 125, 70],
        roughness: 0.85,
        metallic: 0.0,
        emission: 0.0,
    };

    /// Default snow material
    pub const SNOW: Self = Self {
        color: [236, 238, 245],
        roughness: 0.95,
        metallic: 0.0,
        emission: 0.0,
    };

    /// Default sand material
    pub const SAND: Self = Self {
        color: [215, 199, 133],
        roughness: 0.92,
        metallic: 0.0,
        emission: 0.0,
    };

    /// Default water material
    pub const WATER: Self = Self {
        color: [58, 103, 178],
        roughness: 0.4,
        metallic: 0.0,
        emission: 0.0,
    };

    /// Default log material
    pub const LOG: Self = Self {
        color: [94, 68, 42],
        roughness: 0.88,
        metallic: 0.0,
        emission: 0.0,
    };

    /// Default leaves material
    pub const LEAVES: Self = Self {
        color: [62, 114, 52],
        roughness: 0.95,
        metallic: 0.0,
        emission: 0.0,
    };

    /// Default flower material
    pub const FLOWER: Self = Self {
        color: [222, 72, 84],
        roughness: 0.8,
        metallic: 0.0,
        emission: 0.0,
    };
}

/// A single voxel with block type and optional metadata.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct Voxel {
    /// The block type
    pub block_id: BlockId,
    /// Additional metadata (orientation, state, etc.)
    pub metadata: u16,
}

impl Voxel {
    /// Air voxel
    pub const AIR: Self = Self {
        block_id: BlockId::AIR,
        metadata: 0,
    };

    /// Create a new voxel with the given block ID
    #[inline]
    pub const fn new(block_id: BlockId) -> Self {
        Self {
            block_id,
            metadata: 0,
        }
    }

    /// Create a new voxel with block ID and metadata
    #[inline]
    pub const fn with_metadata(block_id: BlockId, metadata: u16) -> Self {
        Self { block_id, metadata }
    }

    /// Returns true if this voxel is air
    #[inline]
    pub const fn is_air(&self) -> bool {
        self.block_id.is_air()
    }

    /// Returns true if this voxel is solid
    #[inline]
    pub const fn is_solid(&self) -> bool {
        self.block_id.is_solid()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_id_air() {
        assert!(BlockId::AIR.is_air());
        assert!(!BlockId::AIR.is_solid());
    }

    #[test]
    fn block_id_solid() {
        assert!(!BlockId::STONE.is_air());
        assert!(BlockId::STONE.is_solid());
        assert!(BlockId::WATER.is_solid());
    }

    #[test]
    fn voxel_default_is_air() {
        let voxel = Voxel::default();
        assert!(voxel.is_air());
    }
}
