//! Clipmap voxel data structures for the Voxelicous engine.

pub mod clipmap;

pub use clipmap::{
    compute_occupancy, decode_brick, downsample_volume_2x, downsample_voxel, encode_brick,
    BrickEncoding, BrickHeader, BrickId, ClipmapPage, ClipmapVoxelStore, EncodedBrick, LodLevel,
    PageId, VoxelCoord, WorldCoord, BRICK_SIZE, BRICK_VOXELS, CLIPMAP_LOD_COUNT, CLIPMAP_PAGE_GRID,
    PAGE_BRICKS, PAGE_BRICKS_PER_AXIS, PAGE_VOXELS_PER_AXIS, PALETTE16_STRIDE, PALETTE32_STRIDE,
    RAW16_STRIDE,
};
