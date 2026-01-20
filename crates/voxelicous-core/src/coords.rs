//! Coordinate systems for the voxel world.

use crate::constants::{CHUNK_BITS, CHUNK_SIZE};
use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Vec3};
use serde::{Deserialize, Serialize};

/// Position within a chunk (0 to CHUNK_SIZE-1 per axis).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable, Serialize, Deserialize,
)]
#[repr(C)]
pub struct LocalPos {
    pub x: u8,
    pub y: u8,
    pub z: u8,
    pub _pad: u8,
}

impl LocalPos {
    /// Create a new local position
    #[inline]
    pub const fn new(x: u8, y: u8, z: u8) -> Self {
        debug_assert!((x as usize) < CHUNK_SIZE);
        debug_assert!((y as usize) < CHUNK_SIZE);
        debug_assert!((z as usize) < CHUNK_SIZE);
        Self { x, y, z, _pad: 0 }
    }

    /// Convert to linear index for flat array storage
    #[inline]
    pub const fn to_index(self) -> usize {
        self.x as usize
            + (self.y as usize) * CHUNK_SIZE
            + (self.z as usize) * CHUNK_SIZE * CHUNK_SIZE
    }

    /// Create from linear index
    #[inline]
    pub const fn from_index(index: usize) -> Self {
        let x = (index % CHUNK_SIZE) as u8;
        let y = ((index / CHUNK_SIZE) % CHUNK_SIZE) as u8;
        let z = (index / (CHUNK_SIZE * CHUNK_SIZE)) as u8;
        Self { x, y, z, _pad: 0 }
    }

    /// Convert to octree path (sequence of child indices)
    #[inline]
    pub fn to_octree_path(self, depth: u32) -> impl Iterator<Item = u8> {
        (0..depth).rev().map(move |level| {
            let shift = level;
            let x_bit = ((self.x as u32 >> shift) & 1) as u8;
            let y_bit = ((self.y as u32 >> shift) & 1) as u8;
            let z_bit = ((self.z as u32 >> shift) & 1) as u8;
            x_bit | (y_bit << 1) | (z_bit << 2)
        })
    }
}

/// Chunk position in chunk coordinates.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable, Serialize, Deserialize,
)]
#[repr(C)]
pub struct ChunkPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub _pad: i32,
}

impl ChunkPos {
    /// Create a new chunk position
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z, _pad: 0 }
    }

    /// Convert to world position (corner of chunk)
    #[inline]
    pub const fn to_world_pos(self) -> WorldPos {
        WorldPos::new(
            (self.x as i64) << CHUNK_BITS,
            (self.y as i64) << CHUNK_BITS,
            (self.z as i64) << CHUNK_BITS,
        )
    }

    /// Get the six neighboring chunk positions
    pub fn neighbors(self) -> [ChunkPos; 6] {
        [
            ChunkPos::new(self.x - 1, self.y, self.z),
            ChunkPos::new(self.x + 1, self.y, self.z),
            ChunkPos::new(self.x, self.y - 1, self.z),
            ChunkPos::new(self.x, self.y + 1, self.z),
            ChunkPos::new(self.x, self.y, self.z - 1),
            ChunkPos::new(self.x, self.y, self.z + 1),
        ]
    }

    /// Convert to glam IVec3
    #[inline]
    pub const fn to_ivec3(self) -> IVec3 {
        IVec3::new(self.x, self.y, self.z)
    }
}

impl From<IVec3> for ChunkPos {
    fn from(v: IVec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }
}

/// World position in voxel coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorldPos {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

impl WorldPos {
    /// Create a new world position
    #[inline]
    pub const fn new(x: i64, y: i64, z: i64) -> Self {
        Self { x, y, z }
    }

    /// Get the chunk containing this position
    #[inline]
    pub const fn chunk_pos(self) -> ChunkPos {
        ChunkPos::new(
            (self.x >> CHUNK_BITS) as i32,
            (self.y >> CHUNK_BITS) as i32,
            (self.z >> CHUNK_BITS) as i32,
        )
    }

    /// Get the local position within the chunk
    #[inline]
    pub const fn local_pos(self) -> LocalPos {
        let mask = (CHUNK_SIZE - 1) as i64;
        LocalPos::new(
            (self.x & mask) as u8,
            (self.y & mask) as u8,
            (self.z & mask) as u8,
        )
    }

    /// Split into chunk and local position
    #[inline]
    pub const fn split(self) -> (ChunkPos, LocalPos) {
        (self.chunk_pos(), self.local_pos())
    }

    /// Create from chunk and local position
    #[inline]
    pub const fn from_chunk_local(chunk: ChunkPos, local: LocalPos) -> Self {
        Self::new(
            ((chunk.x as i64) << CHUNK_BITS) + local.x as i64,
            ((chunk.y as i64) << CHUNK_BITS) + local.y as i64,
            ((chunk.z as i64) << CHUNK_BITS) + local.z as i64,
        )
    }

    /// Convert to floating point Vec3
    #[inline]
    pub fn to_vec3(self) -> Vec3 {
        Vec3::new(self.x as f32, self.y as f32, self.z as f32)
    }
}

impl From<Vec3> for WorldPos {
    fn from(v: Vec3) -> Self {
        Self::new(v.x.floor() as i64, v.y.floor() as i64, v.z.floor() as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_pos_index_roundtrip() {
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let pos = LocalPos::new(x as u8, y as u8, z as u8);
                    let index = pos.to_index();
                    let recovered = LocalPos::from_index(index);
                    assert_eq!(pos, recovered);
                }
            }
        }
    }

    #[test]
    fn world_pos_chunk_local_roundtrip() {
        let world = WorldPos::new(100, -50, 200);
        let (chunk, local) = world.split();
        let recovered = WorldPos::from_chunk_local(chunk, local);
        assert_eq!(world, recovered);
    }

    #[test]
    fn negative_world_pos_chunk() {
        let world = WorldPos::new(-1, -1, -1);
        let chunk = world.chunk_pos();
        assert_eq!(chunk.x, -1);
        assert_eq!(chunk.y, -1);
        assert_eq!(chunk.z, -1);
    }
}
