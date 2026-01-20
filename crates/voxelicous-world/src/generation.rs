//! Procedural terrain generation.

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use rayon::prelude::*;
use voxelicous_core::constants::{CHUNK_SIZE, OCTREE_DEPTH};
use voxelicous_core::coords::ChunkPos;
use voxelicous_core::types::BlockId;
use voxelicous_voxel::{SparseVoxelOctree, VoxelStorage};

use crate::WorldSeed;

/// Terrain generator configuration.
#[derive(Debug, Clone)]
pub struct TerrainConfig {
    /// Seed for noise generation.
    pub seed: WorldSeed,
    /// Sea level (Y coordinate).
    pub sea_level: i32,
    /// Horizontal scale of terrain features.
    pub terrain_scale: f64,
    /// Maximum terrain height variation.
    pub terrain_height: f64,
    /// Number of noise octaves for detail.
    pub octaves: usize,
    /// Frequency multiplier between octaves.
    pub lacunarity: f64,
    /// Amplitude multiplier between octaves.
    pub persistence: f64,
    /// Depth of dirt layer below surface.
    pub dirt_depth: u32,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            sea_level: 64,
            terrain_scale: 100.0,
            terrain_height: 64.0,
            octaves: 4,
            lacunarity: 2.0,
            persistence: 0.5,
            dirt_depth: 4,
        }
    }
}

/// Procedural terrain generator using fractal noise.
pub struct TerrainGenerator {
    config: TerrainConfig,
    height_noise: Fbm<Perlin>,
}

impl TerrainGenerator {
    /// Create a new terrain generator with the given configuration.
    pub fn new(config: TerrainConfig) -> Self {
        let height_noise = Fbm::<Perlin>::new(config.seed as u32)
            .set_octaves(config.octaves)
            .set_lacunarity(config.lacunarity)
            .set_persistence(config.persistence);

        Self {
            config,
            height_noise,
        }
    }

    /// Create a terrain generator with default configuration.
    pub fn with_seed(seed: WorldSeed) -> Self {
        Self::new(TerrainConfig {
            seed,
            ..Default::default()
        })
    }

    /// Get the terrain configuration.
    pub fn config(&self) -> &TerrainConfig {
        &self.config
    }

    /// Get terrain height at world XZ coordinates.
    ///
    /// Returns the Y coordinate of the surface at this position.
    pub fn height_at(&self, world_x: i64, world_z: i64) -> i32 {
        let nx = world_x as f64 / self.config.terrain_scale;
        let nz = world_z as f64 / self.config.terrain_scale;

        // Noise returns [-1, 1], map to [0, terrain_height] and add sea_level
        let noise_value = self.height_noise.get([nx, nz]);
        ((noise_value + 1.0) * 0.5 * self.config.terrain_height) as i32 + self.config.sea_level
    }

    /// Determine block type at a given world Y relative to surface height.
    fn block_at_depth(&self, world_y: i32, surface_height: i32) -> BlockId {
        if world_y > surface_height {
            BlockId::AIR
        } else if world_y == surface_height {
            BlockId::GRASS
        } else if world_y > surface_height - self.config.dirt_depth as i32 {
            BlockId::DIRT
        } else {
            BlockId::STONE
        }
    }

    /// Generate a chunk's voxel data at the given position.
    pub fn generate_chunk(&self, pos: ChunkPos) -> SparseVoxelOctree {
        let mut svo = SparseVoxelOctree::new(OCTREE_DEPTH);
        let world_base = pos.to_world_pos();

        for lz in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let world_x = world_base.x + lx as i64;
                let world_z = world_base.z + lz as i64;
                let surface_height = self.height_at(world_x, world_z);

                for ly in 0..CHUNK_SIZE {
                    let world_y = world_base.y + ly as i64;
                    let block = self.block_at_depth(world_y as i32, surface_height);

                    if block != BlockId::AIR {
                        svo.set(lx as u32, ly as u32, lz as u32, block);
                    }
                }
            }
        }

        svo
    }

    /// Generate multiple chunks in parallel.
    ///
    /// Returns a vector of (position, SVO) pairs.
    pub fn generate_chunks_parallel(
        &self,
        positions: &[ChunkPos],
    ) -> Vec<(ChunkPos, SparseVoxelOctree)> {
        positions
            .par_iter()
            .map(|&pos| (pos, self.generate_chunk(pos)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxelicous_voxel::VoxelStorage;

    #[test]
    fn generator_deterministic() {
        let gen1 = TerrainGenerator::with_seed(12345);
        let gen2 = TerrainGenerator::with_seed(12345);

        // Same seed should produce same heights
        for x in -100..100 {
            for z in -100..100 {
                assert_eq!(gen1.height_at(x, z), gen2.height_at(x, z));
            }
        }
    }

    #[test]
    fn different_seeds_different_terrain() {
        let gen1 = TerrainGenerator::with_seed(12345);
        let gen2 = TerrainGenerator::with_seed(54321);

        // Different seeds should produce different heights (statistically)
        let mut differences = 0;
        for x in 0..10 {
            for z in 0..10 {
                if gen1.height_at(x, z) != gen2.height_at(x, z) {
                    differences += 1;
                }
            }
        }
        assert!(differences > 50, "Seeds should produce different terrain");
    }

    #[test]
    fn chunk_generation() {
        let gen = TerrainGenerator::with_seed(42);
        let chunk = gen.generate_chunk(ChunkPos::new(0, 2, 0)); // Y=2 means world Y 64-95

        // Chunk at sea level should have some terrain
        assert!(!chunk.is_empty());
    }

    #[test]
    fn chunk_above_terrain_is_mostly_empty() {
        let gen = TerrainGenerator::with_seed(42);
        // Y=10 means world Y 320-351, well above terrain
        let chunk = gen.generate_chunk(ChunkPos::new(0, 10, 0));

        // Should be completely empty (all air)
        assert!(chunk.is_empty());
    }

    #[test]
    fn chunk_below_surface_is_solid() {
        let gen = TerrainGenerator::with_seed(42);
        // Y=-2 means world Y -64 to -33, below sea level
        let chunk = gen.generate_chunk(ChunkPos::new(0, -2, 0));

        // Should be solid stone
        assert!(!chunk.is_empty());

        // Check a few voxels
        assert_eq!(chunk.get(0, 0, 0), BlockId::STONE);
        assert_eq!(chunk.get(16, 16, 16), BlockId::STONE);
    }

    #[test]
    fn parallel_generation_matches_sequential() {
        let gen = TerrainGenerator::with_seed(42);
        let positions = vec![
            ChunkPos::new(0, 0, 0),
            ChunkPos::new(1, 0, 0),
            ChunkPos::new(0, 0, 1),
        ];

        let parallel_results = gen.generate_chunks_parallel(&positions);

        for (pos, parallel_svo) in parallel_results {
            let sequential_svo = gen.generate_chunk(pos);

            // Check some sample voxels match
            for x in [0, 15, 31] {
                for y in [0, 15, 31] {
                    for z in [0, 15, 31] {
                        assert_eq!(
                            parallel_svo.get(x, y, z),
                            sequential_svo.get(x, y, z),
                            "Mismatch at ({x}, {y}, {z}) in chunk {:?}",
                            pos
                        );
                    }
                }
            }
        }
    }
}
