//! Procedural terrain generation.

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use voxelicous_core::types::BlockId;

use crate::WorldSeed;

const TREE_CELL_SIZE: i64 = 8;
const TREE_MAX_CANOPY_RADIUS: i64 = 3;

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
    /// Depth of dirt-like layer below surface.
    pub dirt_depth: u32,
    /// Horizontal scale of broad biome regions.
    pub biome_scale: f64,
    /// Relative vertical scale of flatland biomes.
    pub flat_height_scale: f64,
    /// Relative vertical scale of mountain biomes.
    pub mountain_height_scale: f64,
    /// Height above sea level where snow starts appearing.
    pub snow_height_offset: i32,
    /// Randomized offset for snow line in world units.
    pub snow_line_variation: f64,
    /// Horizontal scale of temperature bands.
    pub temperature_scale: f64,
    /// Horizontal scale of moisture bands.
    pub moisture_scale: f64,
    /// Horizontal scale for inland lake placement.
    pub lake_scale: f64,
    /// Noise threshold for placing inland lakes.
    pub lake_threshold: f64,
    /// Horizontal scale of mountain region masks (controls massif size/separation).
    pub mountain_region_scale: f64,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            sea_level: 60,
            terrain_scale: 102.0,
            terrain_height: 72.0,
            octaves: 4,
            lacunarity: 2.0,
            persistence: 0.5,
            dirt_depth: 4,
            biome_scale: 2200.0,
            flat_height_scale: 0.14,
            mountain_height_scale: 1.45,
            snow_height_offset: 44,
            snow_line_variation: 10.0,
            temperature_scale: 2400.0,
            moisture_scale: 2300.0,
            lake_scale: 360.0,
            lake_threshold: 0.56,
            mountain_region_scale: 1900.0,
        }
    }
}

/// Dominant biome at a world XZ coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainBiome {
    /// Mild relief, sparse trees.
    Plains,
    /// Dense vegetation and more frequent trees.
    Forest,
    /// Sandy dry biome.
    Desert,
    /// Rolling or rocky elevated terrain.
    Hills,
    /// High cold peaks with snow.
    SnowyMountains,
}

/// Surface sample for one world XZ column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceSample {
    /// Y coordinate of the terrain surface.
    pub surface_height: i32,
    /// Top block at the surface.
    pub top_block: BlockId,
    /// Block right below surface.
    pub subsurface_block: BlockId,
    /// Dominant biome at this location.
    pub biome: TerrainBiome,
    /// Water level for this column (sea and lakes).
    pub water_level: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TreePlacement {
    pub root_x: i64,
    pub root_z: i64,
    pub trunk_base_y: i32,
    pub trunk_height: i32,
    pub canopy_radius: i32,
}

/// Procedural terrain generator using fractal noise.
#[derive(Clone)]
pub struct TerrainGenerator {
    config: TerrainConfig,
    height_noise: Fbm<Perlin>,
    detail_noise: Fbm<Perlin>,
    ridge_noise: Fbm<Perlin>,
    temperature_noise: Fbm<Perlin>,
    moisture_noise: Fbm<Perlin>,
    desert_noise: Fbm<Perlin>,
    mountain_region_noise: Fbm<Perlin>,
    lake_noise: Fbm<Perlin>,
    lake_depth_noise: Fbm<Perlin>,
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
        let ridge_noise = Fbm::<Perlin>::new(config.seed.wrapping_add(0xC2B2_AE35) as u32)
            .set_octaves(config.octaves + 1)
            .set_lacunarity(config.lacunarity)
            .set_persistence((config.persistence * 0.8).clamp(0.1, 0.95));
        let temperature_noise = Fbm::<Perlin>::new(config.seed.wrapping_add(0xA5A5_5A5A) as u32)
            .set_octaves(2)
            .set_lacunarity(2.0)
            .set_persistence(0.5);
        let moisture_noise = Fbm::<Perlin>::new(config.seed.wrapping_add(0xD6E8_FEB8) as u32)
            .set_octaves(2)
            .set_lacunarity(2.1)
            .set_persistence(0.5);
        let desert_noise = Fbm::<Perlin>::new(config.seed.wrapping_add(0x3C6E_F372) as u32)
            .set_octaves(2)
            .set_lacunarity(1.9)
            .set_persistence(0.5);
        let mountain_region_noise =
            Fbm::<Perlin>::new(config.seed.wrapping_add(0xE703_7ED1) as u32)
                .set_octaves(2)
                .set_lacunarity(1.95)
                .set_persistence(0.5);
        let lake_noise = Fbm::<Perlin>::new(config.seed.wrapping_add(0x94D0_49BB) as u32)
            .set_octaves(2)
            .set_lacunarity(2.0)
            .set_persistence(0.55);
        let lake_depth_noise = Fbm::<Perlin>::new(config.seed.wrapping_add(0xB529_7A4D) as u32)
            .set_octaves(2)
            .set_lacunarity(2.3)
            .set_persistence(0.5);
        let snow_noise = Fbm::<Perlin>::new(config.seed.wrapping_add(0x27D4_EB2F) as u32)
            .set_octaves(2)
            .set_lacunarity(2.0)
            .set_persistence(0.5);

        Self {
            config,
            height_noise,
            detail_noise,
            ridge_noise,
            temperature_noise,
            moisture_noise,
            desert_noise,
            mountain_region_noise,
            lake_noise,
            lake_depth_noise,
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
    pub fn height_at(&self, world_x: i64, world_z: i64) -> i32 {
        self.surface_at(world_x, world_z).surface_height
    }

    /// Get the dominant biome at world XZ coordinates.
    pub fn biome_at(&self, world_x: i64, world_z: i64) -> TerrainBiome {
        self.surface_at(world_x, world_z).biome
    }

    /// Sample the terrain surface at world XZ coordinates.
    pub fn surface_at(&self, world_x: i64, world_z: i64) -> SurfaceSample {
        let nx = world_x as f64 / self.config.terrain_scale;
        let nz = world_z as f64 / self.config.terrain_scale;
        let biome_nx = world_x as f64 / self.config.biome_scale;
        let biome_nz = world_z as f64 / self.config.biome_scale;

        let base = self.height_noise.get([nx * 0.58, nz * 0.58]);
        let detail = self.detail_noise.get([nx * 0.84, nz * 0.84]);
        let micro = self.detail_noise.get([nx * 1.75 + 19.2, nz * 1.75 - 11.8]);
        let ridge = (1.0 - self.ridge_noise.get([nx * 0.55, nz * 0.55]).abs())
            .clamp(0.0, 1.0)
            .powf(0.85);

        let macro_shape = self.height_noise.get([biome_nx * 0.9, biome_nz * 0.9]);
        let macro01 = (macro_shape + 1.0) * 0.5;
        let relief = (ridge * 0.34 + macro01 * 0.66).clamp(0.0, 1.0);
        let mountain_region = (self.mountain_region_noise.get([
            world_x as f64 / self.config.mountain_region_scale,
            world_z as f64 / self.config.mountain_region_scale,
        ]) + 1.0)
            * 0.5;
        let mountain_gate = smoothstep(0.48, 0.74, mountain_region);
        let massif = smoothstep(0.42, 0.80, mountain_region);
        let ridge_break = smoothstep(
            0.68,
            0.90,
            self.detail_noise
                .get([nx * 0.34 + 7.7, nz * 0.34 - 15.3])
                .abs(),
        );
        let mountain_core = smoothstep(0.66, 0.88, relief);
        let mountain_weight = mountain_core
            * mountain_gate
            * (1.0 - ridge_break * 0.28).clamp(0.0, 1.0)
            * (0.85 + 0.15 * massif);
        let hill_weight = smoothstep(0.48, 0.82, relief) * (1.0 - mountain_weight * 0.82);
        let flat_weight = (1.0 - hill_weight).clamp(0.0, 1.0) * (1.0 - mountain_weight);

        let flat_component = (base * 0.70 + detail * 0.22 + micro * 0.14)
            * (self.config.terrain_height * self.config.flat_height_scale);
        let hill_component =
            (base * 0.76 + detail * 0.20 + micro * 0.10) * (self.config.terrain_height * 0.52);
        let mountain_body =
            (base * 0.78 + detail * 0.12 + micro * 0.05) * (self.config.terrain_height * 0.88);
        let mountain_ridge =
            ridge * self.config.terrain_height * self.config.mountain_height_scale * 0.55;
        let massif_uplift = (massif - 0.35).max(0.0) * self.config.terrain_height * 0.55;
        let mountain_component = mountain_body + mountain_ridge + massif_uplift;

        let mut height_offset = flat_component * flat_weight
            + hill_component * hill_weight
            + mountain_component * mountain_weight;
        height_offset +=
            micro * (self.config.terrain_height * 0.12) * (1.0 - mountain_weight * 0.40);
        let shoulder = smoothstep(0.44, 0.78, mountain_weight) * massif;
        height_offset += shoulder * self.config.terrain_height * 0.22;

        let highland_factor = smoothstep(0.44, 0.78, mountain_weight) * massif;
        if highland_factor > 0.0 {
            let plateau_step = 6.0;
            let snapped = (height_offset / plateau_step).round() * plateau_step;
            let plateau_mix = (highland_factor * 0.76).clamp(0.0, 0.76);
            height_offset = height_offset * (1.0 - plateau_mix) + snapped * plateau_mix;
        }
        let surface_height = self.config.sea_level + height_offset.round() as i32;

        let temperature = self.temperature_noise.get([
            world_x as f64 / self.config.temperature_scale,
            world_z as f64 / self.config.temperature_scale,
        ]);
        let moisture = self.moisture_noise.get([
            world_x as f64 / self.config.moisture_scale,
            world_z as f64 / self.config.moisture_scale,
        ]);
        let desert_region = self
            .desert_noise
            .get([biome_nx * 0.65 + 7.1, biome_nz * 0.65 - 9.3]);

        let biome = dominant_biome(
            mountain_weight,
            hill_weight,
            temperature,
            moisture,
            desert_region,
            surface_height,
            self.config.sea_level + self.config.snow_height_offset,
        );
        let snow_line_noise = self.snow_noise.get([nx * 0.7 + 13.7, nz * 0.7 - 21.3])
            * self.config.snow_line_variation;
        let snow_line = self.config.sea_level + self.config.snow_height_offset;
        let snow_threshold = f64::from(snow_line) + snow_line_noise;
        let (top_block, subsurface_block) = surface_blocks_for_biome(
            biome,
            ridge,
            surface_height,
            self.config.sea_level,
            snow_threshold,
        );
        let water_level = self.water_level_at(
            world_x,
            world_z,
            surface_height,
            biome,
            self.config.sea_level,
        );

        SurfaceSample {
            surface_height,
            top_block,
            subsurface_block,
            biome,
            water_level,
        }
    }

    pub(crate) fn block_from_surface_sample(
        &self,
        world_x: i64,
        world_y: i64,
        world_z: i64,
        surface: SurfaceSample,
    ) -> BlockId {
        let world_y = clamp_i64_to_i32(world_y);
        if world_y > surface.surface_height {
            if world_y <= surface.water_level {
                BlockId::WATER
            } else if self.should_place_flower(world_x, world_z, surface) {
                if world_y == surface.surface_height + 1 {
                    BlockId::FLOWER
                } else {
                    BlockId::AIR
                }
            } else {
                BlockId::AIR
            }
        } else if world_y == surface.surface_height {
            surface.top_block
        } else if world_y > surface.surface_height - self.config.dirt_depth as i32 {
            surface.subsurface_block
        } else {
            BlockId::STONE
        }
    }

    pub(crate) fn trees_in_area(
        &self,
        min_x: i64,
        max_x: i64,
        min_z: i64,
        max_z: i64,
    ) -> Vec<TreePlacement> {
        let mut out = Vec::new();
        let cell_min_x = div_floor(min_x - TREE_MAX_CANOPY_RADIUS, TREE_CELL_SIZE);
        let cell_max_x = div_floor(max_x + TREE_MAX_CANOPY_RADIUS, TREE_CELL_SIZE);
        let cell_min_z = div_floor(min_z - TREE_MAX_CANOPY_RADIUS, TREE_CELL_SIZE);
        let cell_max_z = div_floor(max_z + TREE_MAX_CANOPY_RADIUS, TREE_CELL_SIZE);

        for cell_z in cell_min_z..=cell_max_z {
            for cell_x in cell_min_x..=cell_max_x {
                let Some(tree) = self.tree_in_cell(cell_x, cell_z) else {
                    continue;
                };
                let radius = i64::from(tree.canopy_radius);
                if tree.root_x + radius < min_x
                    || tree.root_x - radius > max_x
                    || tree.root_z + radius < min_z
                    || tree.root_z - radius > max_z
                {
                    continue;
                }
                out.push(tree);
            }
        }

        out
    }

    pub(crate) fn tree_block_for_placement(
        tree: TreePlacement,
        world_x: i64,
        world_y: i64,
        world_z: i64,
    ) -> Option<BlockId> {
        let world_y = clamp_i64_to_i32(world_y);
        let trunk_top = tree.trunk_base_y + tree.trunk_height;
        if world_x == tree.root_x
            && world_z == tree.root_z
            && world_y >= tree.trunk_base_y
            && world_y < trunk_top
        {
            return Some(BlockId::LOG);
        }

        let center_y = trunk_top - 1;
        let dx = (world_x - tree.root_x).abs() as i32;
        let dz = (world_z - tree.root_z).abs() as i32;
        let dy = (world_y - center_y).abs();
        if dy > 2 {
            return None;
        }
        if dx > tree.canopy_radius || dz > tree.canopy_radius {
            return None;
        }

        let radius_sq = tree.canopy_radius * tree.canopy_radius + 1;
        let dist_sq = dx * dx + dz * dz + dy * dy * 2;
        if dist_sq <= radius_sq {
            Some(BlockId::LEAVES)
        } else {
            None
        }
    }

    fn tree_block_at(&self, world_x: i64, world_y: i64, world_z: i64) -> Option<BlockId> {
        let cell_x = div_floor(world_x, TREE_CELL_SIZE);
        let cell_z = div_floor(world_z, TREE_CELL_SIZE);
        let mut leaf_hit = None;
        for dz in -1..=1 {
            for dx in -1..=1 {
                let Some(tree) = self.tree_in_cell(cell_x + dx, cell_z + dz) else {
                    continue;
                };
                if let Some(block) = Self::tree_block_for_placement(tree, world_x, world_y, world_z)
                {
                    if block == BlockId::LOG {
                        return Some(BlockId::LOG);
                    }
                    if leaf_hit.is_none() {
                        leaf_hit = Some(block);
                    }
                }
            }
        }
        leaf_hit
    }

    fn tree_in_cell(&self, cell_x: i64, cell_z: i64) -> Option<TreePlacement> {
        let hash = hash2(self.config.seed.wrapping_add(0x6C8E_9CF5), cell_x, cell_z);
        let inner = (TREE_CELL_SIZE - 2) as u64;
        let offset_x = 1 + ((hash >> 8) % inner) as i64;
        let offset_z = 1 + ((hash >> 16) % inner) as i64;
        let root_x = cell_x * TREE_CELL_SIZE + offset_x;
        let root_z = cell_z * TREE_CELL_SIZE + offset_z;

        let surface = self.surface_at(root_x, root_z);
        if surface.top_block != BlockId::GRASS {
            return None;
        }
        if surface.water_level > surface.surface_height {
            return None;
        }

        let density = match surface.biome {
            TerrainBiome::Forest => 0.26,
            TerrainBiome::Plains => 0.07,
            TerrainBiome::Hills => 0.03,
            TerrainBiome::Desert | TerrainBiome::SnowyMountains => 0.0,
        };
        if density <= 0.0 || hash_to_unit(hash >> 24) >= density {
            return None;
        }

        let trunk_height = 4 + ((hash >> 32) % 3) as i32;
        let canopy_radius = 2 + ((hash >> 40) % 2) as i32;
        Some(TreePlacement {
            root_x,
            root_z,
            trunk_base_y: surface.surface_height + 1,
            trunk_height,
            canopy_radius,
        })
    }

    fn should_place_flower(&self, world_x: i64, world_z: i64, surface: SurfaceSample) -> bool {
        if surface.top_block != BlockId::GRASS || surface.water_level > surface.surface_height {
            return false;
        }
        let chance = match surface.biome {
            TerrainBiome::Forest => 0.03,
            TerrainBiome::Plains => 0.02,
            TerrainBiome::Desert | TerrainBiome::Hills | TerrainBiome::SnowyMountains => 0.0,
        };
        if chance <= 0.0 {
            return false;
        }
        let hash = hash2(self.config.seed.wrapping_add(0x8B8B_8B8B), world_x, world_z);
        hash_to_unit(hash) < chance
    }

    fn water_level_at(
        &self,
        world_x: i64,
        world_z: i64,
        surface_height: i32,
        biome: TerrainBiome,
        sea_level: i32,
    ) -> i32 {
        let mut water_level = sea_level;
        if matches!(
            biome,
            TerrainBiome::Desert | TerrainBiome::Hills | TerrainBiome::SnowyMountains
        ) || surface_height <= sea_level
        {
            return water_level;
        }

        // Avoid elevated "floating" lakes by restricting lakes to lower plains/forest regions.
        if surface_height > sea_level + 14 {
            return water_level;
        }

        let lake_nx = world_x as f64 / self.config.lake_scale;
        let lake_nz = world_z as f64 / self.config.lake_scale;
        let lake_mask = self.lake_noise.get([lake_nx, lake_nz]);
        if lake_mask <= self.config.lake_threshold {
            return water_level;
        }

        let basin_factor = ((lake_mask - self.config.lake_threshold)
            / (1.0 - self.config.lake_threshold))
            .clamp(0.0, 1.0);
        if basin_factor < 0.22 {
            return water_level;
        }

        let lake_depth01 = (self
            .lake_depth_noise
            .get([lake_nx * 1.7 + 8.3, lake_nz * 1.7 - 4.1])
            + 1.0)
            * 0.5;
        let local_lake_level = sea_level + 1 + (lake_depth01 * 4.0).round() as i32;
        let lake_depth = local_lake_level - surface_height;
        if lake_depth > 0 && lake_depth <= 4 {
            water_level = water_level.max(local_lake_level);
        }

        water_level
    }

    /// Get block ID at world coordinates.
    pub fn block_at_world(&self, world_x: i64, world_y: i64, world_z: i64) -> BlockId {
        let surface = self.surface_at(world_x, world_z);
        let base_block = self.block_from_surface_sample(world_x, world_y, world_z, surface);
        if base_block == BlockId::AIR || base_block == BlockId::FLOWER {
            if let Some(tree_block) = self.tree_block_at(world_x, world_y, world_z) {
                return tree_block;
            }
        }

        base_block
    }
}

fn surface_blocks_for_biome(
    biome: TerrainBiome,
    ridge: f64,
    surface_height: i32,
    sea_level: i32,
    snow_threshold: f64,
) -> (BlockId, BlockId) {
    if surface_height <= sea_level + 1 {
        return (BlockId::SAND, BlockId::SAND);
    }

    match biome {
        TerrainBiome::Desert => (BlockId::SAND, BlockId::SAND),
        TerrainBiome::SnowyMountains => {
            if f64::from(surface_height) >= snow_threshold {
                (BlockId::SNOW, BlockId::STONE)
            } else {
                (BlockId::STONE, BlockId::STONE)
            }
        }
        TerrainBiome::Hills => {
            if ridge > 0.93 && surface_height > sea_level + 34 {
                (BlockId::STONE, BlockId::STONE)
            } else if ridge > 0.86 && surface_height > sea_level + 26 {
                (BlockId::GRASS, BlockId::STONE)
            } else {
                (BlockId::GRASS, BlockId::DIRT)
            }
        }
        TerrainBiome::Plains | TerrainBiome::Forest => (BlockId::GRASS, BlockId::DIRT),
    }
}

fn dominant_biome(
    mountain_weight: f64,
    hill_weight: f64,
    temperature: f64,
    moisture: f64,
    desert_region: f64,
    surface_height: i32,
    snow_line_hint: i32,
) -> TerrainBiome {
    let cold = temperature < -0.12;
    let wet = moisture > 0.22;
    let high_elevation = surface_height >= snow_line_hint - 10;

    if mountain_weight > 0.54 && cold && high_elevation {
        return TerrainBiome::SnowyMountains;
    }
    let heat = smoothstep(0.00, 0.45, temperature);
    let aridity = smoothstep(-0.08, 0.45, temperature - moisture * 0.95);
    let desert_zone = smoothstep(0.08, 0.55, desert_region);
    let desert_strength = (heat * 0.40 + aridity * 0.40 + desert_zone * 0.20).clamp(0.0, 1.0);
    if desert_strength > 0.58 && mountain_weight < 0.58 {
        return TerrainBiome::Desert;
    }
    if mountain_weight > 0.52 || hill_weight > 0.62 {
        return TerrainBiome::Hills;
    }
    if wet {
        TerrainBiome::Forest
    } else {
        TerrainBiome::Plains
    }
}

fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if edge0 == edge1 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn clamp_i64_to_i32(value: i64) -> i32 {
    if value < i64::from(i32::MIN) {
        i32::MIN
    } else if value > i64::from(i32::MAX) {
        i32::MAX
    } else {
        value as i32
    }
}

fn div_floor(value: i64, divisor: i64) -> i64 {
    let mut q = value / divisor;
    let r = value % divisor;
    if r != 0 && ((r > 0) != (divisor > 0)) {
        q -= 1;
    }
    q
}

fn hash2(seed: u64, x: i64, z: i64) -> u64 {
    let mut v = seed
        ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (z as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^ (v >> 31)
}

fn hash_to_unit(hash: u64) -> f64 {
    let mantissa = hash >> 11;
    mantissa as f64 / ((1u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_deterministic() {
        let gen1 = TerrainGenerator::with_seed(12345);
        let gen2 = TerrainGenerator::with_seed(12345);

        for x in -100..100 {
            for z in -100..100 {
                assert_eq!(gen1.height_at(x, z), gen2.height_at(x, z));
                assert_eq!(gen1.surface_at(x, z), gen2.surface_at(x, z));
            }
        }
    }

    #[test]
    fn different_seeds_different_terrain() {
        let gen1 = TerrainGenerator::with_seed(12345);
        let gen2 = TerrainGenerator::with_seed(54321);

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
        let x = 96;
        let z = -41;
        let surface = generator.surface_at(x, z);

        assert_eq!(
            generator.block_at_world(x, i64::from(surface.surface_height), z),
            surface.top_block
        );
        assert_eq!(
            generator.block_at_world(x, i64::from(surface.surface_height - 32), z),
            BlockId::STONE
        );

        let above_surface = generator.block_at_world(x, i64::from(surface.surface_height + 1), z);
        if surface.water_level > surface.surface_height {
            assert_eq!(above_surface, BlockId::WATER);
        } else {
            assert_ne!(above_surface, BlockId::WATER);
        }
    }

    #[test]
    fn world_contains_multiple_biomes() {
        let generator = TerrainGenerator::with_seed(12345);
        let mut saw_plains = false;
        let mut saw_forest = false;
        let mut saw_desert = false;
        let mut saw_hills = false;
        let mut saw_snowy = false;

        for x in (-8192..=8192).step_by(128) {
            for z in (-8192..=8192).step_by(128) {
                match generator.biome_at(x, z) {
                    TerrainBiome::Plains => saw_plains = true,
                    TerrainBiome::Forest => saw_forest = true,
                    TerrainBiome::Desert => saw_desert = true,
                    TerrainBiome::Hills => saw_hills = true,
                    TerrainBiome::SnowyMountains => saw_snowy = true,
                }

                if saw_plains && saw_forest && saw_desert && saw_hills && saw_snowy {
                    break;
                }
            }
            if saw_plains && saw_forest && saw_desert && saw_hills && saw_snowy {
                break;
            }
        }

        assert!(saw_plains, "Expected to encounter plains biome");
        assert!(saw_forest, "Expected to encounter forest biome");
        assert!(saw_desert, "Expected to encounter desert biome");
        assert!(saw_hills, "Expected to encounter hills biome");
        assert!(saw_snowy, "Expected to encounter snowy mountain biome");
    }

    #[test]
    fn snow_appears_on_some_mountain_peaks() {
        let generator = TerrainGenerator::with_seed(42);
        let mut found_snow = false;

        'outer: for x in (-12288..=12288).step_by(64) {
            for z in (-12288..=12288).step_by(64) {
                let sample = generator.surface_at(x, z);
                if sample.biome == TerrainBiome::SnowyMountains && sample.top_block == BlockId::SNOW
                {
                    found_snow = true;
                    break 'outer;
                }
            }
        }

        assert!(found_snow, "Expected to find snow in snowy mountains");
    }

    #[test]
    fn world_contains_water_columns() {
        let generator = TerrainGenerator::with_seed(12345);
        let mut found_water = false;

        'outer: for x in (-2048..=2048).step_by(32) {
            for z in (-2048..=2048).step_by(32) {
                let sample = generator.surface_at(x, z);
                if sample.water_level > sample.surface_height {
                    found_water = true;
                    break 'outer;
                }
            }
        }

        assert!(found_water, "Expected to find sea or lake water columns");
    }

    #[test]
    fn trees_respect_biome_and_water_constraints() {
        let generator = TerrainGenerator::with_seed(42);
        let trees = generator.trees_in_area(-4096, 4096, -4096, 4096);
        assert!(!trees.is_empty(), "Expected at least a few trees");

        for tree in trees {
            let surface = generator.surface_at(tree.root_x, tree.root_z);
            assert_ne!(surface.biome, TerrainBiome::Desert);
            assert_ne!(surface.biome, TerrainBiome::SnowyMountains);
            assert_eq!(surface.top_block, BlockId::GRASS);
            assert!(
                surface.water_level <= surface.surface_height,
                "Trees should not spawn in water"
            );
        }
    }

    #[test]
    fn world_has_meaningful_vertical_relief() {
        let generator = TerrainGenerator::with_seed(42);
        let mut min_h = i32::MAX;
        let mut max_h = i32::MIN;

        for x in (-2048..=2048).step_by(64) {
            for z in (-2048..=2048).step_by(64) {
                let h = generator.height_at(x, z);
                min_h = min_h.min(h);
                max_h = max_h.max(h);
            }
        }

        assert!(
            max_h - min_h >= 30,
            "Expected at least 30 voxels of vertical relief, got {}",
            max_h - min_h
        );
    }

    #[test]
    fn hills_are_not_mostly_stone_on_surface() {
        let generator = TerrainGenerator::with_seed(42);
        let mut hills_total = 0;
        let mut hills_stone_surface = 0;

        for x in (-8192..=8192).step_by(64) {
            for z in (-8192..=8192).step_by(64) {
                let sample = generator.surface_at(x, z);
                if sample.biome == TerrainBiome::Hills {
                    hills_total += 1;
                    if sample.top_block == BlockId::STONE {
                        hills_stone_surface += 1;
                    }
                }
            }
        }

        assert!(hills_total > 0, "Expected to sample some hills");
        assert!(
            hills_stone_surface * 100 <= hills_total * 45,
            "Expected <=45% stone hill tops, got {hills_stone_surface}/{hills_total}"
        );
    }

    #[test]
    fn deserts_have_large_presence() {
        let generator = TerrainGenerator::with_seed(42);
        let mut total = 0usize;
        let mut desert = 0usize;

        for x in (-8192..=8192).step_by(128) {
            for z in (-8192..=8192).step_by(128) {
                total += 1;
                if generator.biome_at(x, z) == TerrainBiome::Desert {
                    desert += 1;
                }
            }
        }

        assert!(
            desert * 100 >= total * 6,
            "Expected >=6% desert coverage, got {desert}/{total}"
        );
    }

    #[test]
    fn tree_trunk_base_overrides_flowers_when_positions_overlap() {
        let mut overlap_case = None;

        'search: for seed in 0..256_u64 {
            let generator = TerrainGenerator::with_seed(seed);
            let trees = generator.trees_in_area(-2048, 2048, -2048, 2048);
            for tree in trees {
                let surface = generator.surface_at(tree.root_x, tree.root_z);
                let trunk_y = i64::from(tree.trunk_base_y);
                let base =
                    generator.block_from_surface_sample(tree.root_x, trunk_y, tree.root_z, surface);
                if base != BlockId::FLOWER {
                    continue;
                }
                overlap_case = Some((seed, tree.root_x, trunk_y, tree.root_z));
                break 'search;
            }
        }

        let Some((seed, root_x, root_y, root_z)) = overlap_case else {
            panic!("Expected at least one flower/tree-root overlap case in sampled seeds");
        };
        let generator = TerrainGenerator::with_seed(seed);
        assert_eq!(
            generator.block_at_world(root_x, root_y, root_z),
            BlockId::LOG,
            "Tree trunk base must override flower at ({root_x}, {root_y}, {root_z})"
        );
    }
}
