//! Terrain generation utilities.

use tracing::info;
use voxelicous_core::coords::ChunkPos;
use voxelicous_core::BlockId;
use voxelicous_voxel::{SparseVoxelOctree, SvoDag, VoxelStorage};
use voxelicous_world::{TerrainConfig, TerrainGenerator};

/// World generation parameters.
pub struct WorldParams {
    pub chunks_x: i32,
    pub chunks_y: i32,
    pub chunks_z: i32,
    pub world_depth: u32,
    pub seed: u64,
}

impl Default for WorldParams {
    fn default() -> Self {
        Self {
            chunks_x: 4,
            chunks_y: 2,
            chunks_z: 4,
            world_depth: 7, // 2^7 = 128, enough for 4 chunks per axis
            seed: 42,
        }
    }
}

/// Generated world data.
pub struct GeneratedWorld {
    pub dag: SvoDag,
    #[allow(dead_code)]
    pub total_voxels: u64,
}

/// Generate terrain and compress it to a DAG.
pub fn generate_world(params: &WorldParams) -> GeneratedWorld {
    info!("Generating terrain...");

    let terrain_config = TerrainConfig {
        seed: params.seed,
        sea_level: 64,
        terrain_scale: 50.0,
        terrain_height: 32.0,
        ..Default::default()
    };
    let generator = TerrainGenerator::new(terrain_config);

    info!(
        "Generating {}x{}x{} chunks ({} total)...",
        params.chunks_x,
        params.chunks_y,
        params.chunks_z,
        params.chunks_x * params.chunks_y * params.chunks_z
    );

    let mut world_svo = SparseVoxelOctree::new(params.world_depth);
    let chunk_size = 32u32;
    let mut total_voxels = 0u64;

    for cy in 0..params.chunks_y {
        for cz in 0..params.chunks_z {
            for cx in 0..params.chunks_x {
                let chunk_pos = ChunkPos::new(cx, cy + 2, cz);
                let chunk_svo = generator.generate_chunk(chunk_pos);

                let base_x = (cx as u32) * chunk_size;
                let base_y = (cy as u32) * chunk_size;
                let base_z = (cz as u32) * chunk_size;

                for z in 0..chunk_size {
                    for y in 0..chunk_size {
                        for x in 0..chunk_size {
                            let block = chunk_svo.get(x, y, z);
                            if block != BlockId::AIR {
                                world_svo.set(base_x + x, base_y + y, base_z + z, block);
                                total_voxels += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    info!("World generated: {} solid voxels", total_voxels);

    info!("Compressing to DAG...");
    let dag = SvoDag::from_svo(&world_svo);
    info!(
        "DAG nodes: {}, compression ratio: {:.2}x",
        dag.node_count(),
        dag.compression_ratio(world_svo.root().count_nodes())
    );

    GeneratedWorld { dag, total_voxels }
}
