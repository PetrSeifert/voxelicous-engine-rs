//! Procedural terrain generation.

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use voxelicous_core::types::BlockId;

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
#[derive(Clone)]
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

    /// Get block ID at world coordinates.
    pub fn block_at_world(&self, world_x: i64, world_y: i64, world_z: i64) -> BlockId {
        let surface_height = self.height_at(world_x, world_z);
        self.block_at_depth(world_y as i32, surface_height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn block_sampling_matches_surface_logic() {
        let generator = TerrainGenerator::with_seed(42);
        let x = 32;
        let z = -24;
        let surface = generator.height_at(x, z);

        assert_eq!(
            generator.block_at_world(x, i64::from(surface), z),
            BlockId::GRASS
        );
        assert_eq!(
            generator.block_at_world(x, i64::from(surface - 1), z),
            BlockId::DIRT
        );
        assert_eq!(
            generator.block_at_world(x, i64::from(surface - 32), z),
            BlockId::STONE
        );
        assert_eq!(
            generator.block_at_world(x, i64::from(surface + 1), z),
            BlockId::AIR
        );
    }
}
