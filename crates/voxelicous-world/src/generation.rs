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
    /// Horizontal scale of biome patches (flatlands, hills, mountains).
    pub biome_scale: f64,
    /// Relative vertical scale of flatland biome.
    pub flat_height_scale: f64,
    /// Relative vertical scale of mountain biome.
    pub mountain_height_scale: f64,
    /// Height above sea level where snow starts appearing on mountains.
    pub snow_height_offset: i32,
    /// Randomized offset for snow line in world units.
    pub snow_line_variation: f64,
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
            biome_scale: 700.0,
            flat_height_scale: 0.10,
            mountain_height_scale: 1.95,
            snow_height_offset: 44,
            snow_line_variation: 10.0,
        }
    }
}

/// Dominant biome at a world XZ coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainBiome {
    /// Mostly low relief.
    Flatlands,
    /// Medium relief rolling terrain.
    Hills,
    /// High relief steep terrain.
    Mountains,
}

/// Surface sample for one world XZ column.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceSample {
    /// Y coordinate of the terrain surface.
    pub surface_height: i32,
    /// Top block at the surface.
    pub top_block: BlockId,
    /// Dominant biome at this location.
    pub biome: TerrainBiome,
}

/// Procedural terrain generator using fractal noise.
#[derive(Clone)]
pub struct TerrainGenerator {
    config: TerrainConfig,
    height_noise: Fbm<Perlin>,
    detail_noise: Fbm<Perlin>,
    biome_noise: Fbm<Perlin>,
    ridge_noise: Fbm<Perlin>,
    snow_noise: Fbm<Perlin>,
}

impl TerrainGenerator {
    /// Create a new terrain generator with the given configuration.
    pub fn new(config: TerrainConfig) -> Self {
        let height_noise = Fbm::<Perlin>::new(config.seed as u32)
            .set_octaves(config.octaves)
            .set_lacunarity(config.lacunarity)
            .set_persistence(config.persistence);
        let detail_noise = Fbm::<Perlin>::new(config.seed.wrapping_add(0x9E37_79B9) as u32)
            .set_octaves(config.octaves.saturating_sub(1).max(1))
            .set_lacunarity(config.lacunarity)
            .set_persistence(config.persistence);
        let biome_noise = Fbm::<Perlin>::new(config.seed.wrapping_add(0xA5A5_5A5A) as u32)
            .set_octaves(2)
            .set_lacunarity(2.0)
            .set_persistence(0.5);
        let ridge_noise = Fbm::<Perlin>::new(config.seed.wrapping_add(0xC2B2_AE35) as u32)
            .set_octaves(config.octaves + 1)
            .set_lacunarity(config.lacunarity)
            .set_persistence((config.persistence * 0.8).clamp(0.1, 0.95));
        let snow_noise = Fbm::<Perlin>::new(config.seed.wrapping_add(0x27D4_EB2F) as u32)
            .set_octaves(2)
            .set_lacunarity(2.0)
            .set_persistence(0.5);

        Self {
            config,
            height_noise,
            detail_noise,
            biome_noise,
            ridge_noise,
            snow_noise,
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
        self.surface_at(world_x, world_z).surface_height
    }

    /// Get the dominant biome at world XZ coordinates.
    pub fn biome_at(&self, world_x: i64, world_z: i64) -> TerrainBiome {
        let biome_nx = world_x as f64 / self.config.biome_scale;
        let biome_nz = world_z as f64 / self.config.biome_scale;
        let biome_value = self.biome_noise.get([biome_nx, biome_nz]);
        let (flat_weight, hill_weight, mountain_weight) = biome_weights(biome_value);
        dominant_biome(flat_weight, hill_weight, mountain_weight)
    }

    /// Sample the terrain surface at world XZ coordinates.
    pub fn surface_at(&self, world_x: i64, world_z: i64) -> SurfaceSample {
        let nx = world_x as f64 / self.config.terrain_scale;
        let nz = world_z as f64 / self.config.terrain_scale;
        let biome_nx = world_x as f64 / self.config.biome_scale;
        let biome_nz = world_z as f64 / self.config.biome_scale;

        // Use lower-frequency signals so terrain is broader and more walkable.
        let base = self.height_noise.get([nx * 0.65, nz * 0.65]);
        let detail = self.detail_noise.get([nx * 0.90, nz * 0.90]);
        let biome_value = self.biome_noise.get([biome_nx, biome_nz]);
        let ridge = (1.0 - self.ridge_noise.get([nx * 0.55, nz * 0.55]).abs())
            .clamp(0.0, 1.0)
            .powf(0.85);
        let (flat_weight, hill_weight, mountain_weight) = biome_weights(biome_value);

        let flat_component = (base * 0.82 + detail * 0.10)
            * (self.config.terrain_height * self.config.flat_height_scale);
        let hill_component = (base * 0.86 + detail * 0.16) * (self.config.terrain_height * 0.46);
        let mountain_component = (base * 0.72 + detail * 0.08)
            * (self.config.terrain_height * 0.68)
            + ridge * self.config.terrain_height * self.config.mountain_height_scale;

        let height_offset = flat_component * flat_weight
            + hill_component * hill_weight
            + mountain_component * mountain_weight;
        let surface_height = self.config.sea_level + height_offset.round() as i32;

        let snow_line_noise = self.snow_noise.get([nx * 0.7 + 13.7, nz * 0.7 - 21.3])
            * self.config.snow_line_variation;
        let snow_line = self.config.sea_level + self.config.snow_height_offset;
        let snow_threshold = f64::from(snow_line) + snow_line_noise;
        let top_block = if mountain_weight > 0.35 && f64::from(surface_height) >= snow_threshold {
            BlockId::SNOW
        } else {
            BlockId::GRASS
        };
        let biome = dominant_biome(flat_weight, hill_weight, mountain_weight);

        SurfaceSample {
            surface_height,
            top_block,
            biome,
        }
    }

    /// Determine block type at a given world Y relative to surface height.
    fn block_at_depth(&self, world_y: i32, surface: SurfaceSample) -> BlockId {
        if world_y > surface.surface_height {
            BlockId::AIR
        } else if world_y == surface.surface_height {
            surface.top_block
        } else if world_y > surface.surface_height - self.config.dirt_depth as i32 {
            BlockId::DIRT
        } else {
            BlockId::STONE
        }
    }

    /// Get block ID at world coordinates.
    pub fn block_at_world(&self, world_x: i64, world_y: i64, world_z: i64) -> BlockId {
        let surface = self.surface_at(world_x, world_z);
        self.block_at_depth(world_y as i32, surface)
    }
}

fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if edge0 == edge1 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn biome_weights(biome_value: f64) -> (f64, f64, f64) {
    let mountain = smoothstep(0.18, 0.62, biome_value);
    let flat = 1.0 - smoothstep(-0.62, -0.18, biome_value);
    let hill = (1.0 - flat - mountain).clamp(0.0, 1.0);
    let sum = flat + hill + mountain;
    if sum <= f64::EPSILON {
        (0.0, 1.0, 0.0)
    } else {
        (flat / sum, hill / sum, mountain / sum)
    }
}

fn dominant_biome(flat_weight: f64, hill_weight: f64, mountain_weight: f64) -> TerrainBiome {
    if mountain_weight >= hill_weight && mountain_weight >= flat_weight {
        TerrainBiome::Mountains
    } else if flat_weight >= hill_weight {
        TerrainBiome::Flatlands
    } else {
        TerrainBiome::Hills
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
        for x in 0..24 {
            for z in 0..24 {
                if gen1.height_at(x, z) != gen2.height_at(x, z) {
                    differences += 1;
                }
            }
        }
        assert!(differences > 150, "Seeds should produce different terrain");
    }

    #[test]
    fn block_sampling_matches_surface_logic() {
        let generator = TerrainGenerator::with_seed(42);
        let x = 32;
        let z = -24;
        let surface = generator.surface_at(x, z);

        assert_eq!(
            generator.block_at_world(x, i64::from(surface.surface_height), z),
            surface.top_block
        );
        assert_eq!(
            generator.block_at_world(x, i64::from(surface.surface_height - 1), z),
            BlockId::DIRT
        );
        assert_eq!(
            generator.block_at_world(x, i64::from(surface.surface_height - 32), z),
            BlockId::STONE
        );
        assert_eq!(
            generator.block_at_world(x, i64::from(surface.surface_height + 1), z),
            BlockId::AIR
        );
    }

    #[test]
    fn world_contains_multiple_biomes() {
        let generator = TerrainGenerator::with_seed(12345);
        let mut saw_flat = false;
        let mut saw_hills = false;
        let mut saw_mountains = false;

        for x in (-2048..=2048).step_by(64) {
            for z in (-2048..=2048).step_by(64) {
                match generator.biome_at(x, z) {
                    TerrainBiome::Flatlands => saw_flat = true,
                    TerrainBiome::Hills => saw_hills = true,
                    TerrainBiome::Mountains => saw_mountains = true,
                }

                if saw_flat && saw_hills && saw_mountains {
                    break;
                }
            }
            if saw_flat && saw_hills && saw_mountains {
                break;
            }
        }

        assert!(saw_flat, "Expected to encounter flatland biome");
        assert!(saw_hills, "Expected to encounter hill biome");
        assert!(saw_mountains, "Expected to encounter mountain biome");
    }

    #[test]
    fn snow_appears_on_some_mountain_peaks() {
        let generator = TerrainGenerator::with_seed(42);
        let mut found_snow = false;

        'outer: for x in (-4096..=4096).step_by(32) {
            for z in (-4096..=4096).step_by(32) {
                let sample = generator.surface_at(x, z);
                if sample.biome == TerrainBiome::Mountains && sample.top_block == BlockId::SNOW {
                    found_snow = true;
                    break 'outer;
                }
            }
        }

        assert!(found_snow, "Expected to find snow in mountain biome");
    }
}
