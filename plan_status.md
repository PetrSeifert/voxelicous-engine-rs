# Plan Status

## Done
- Added clipmap voxel storage with brick headers, palette/raw pools, occupancy masks, and downsampling helpers.
- Implemented `ClipmapStreamingController` with toroidal page tables, LOD origin management, and dirty tracking.
- Added a per-frame clipmap page rebuild budget to avoid full rebuild stalls.
- Added partial GPU buffer writes for incremental updates.
- Implemented clipmap GPU renderer + push constants, plus compute pipeline using `ray_march_clipmap.comp`.
- Added new clipmap shader and wired it into build + shader loader.
- Updated viewer to drive clipmap streaming/rendering.
- Added LOD readiness gating (skip LODs until their page tables are fully built).
- Added DDA boundary epsilon and interval end-guarding to reduce wrap/edge artifacts.
- Increased ray-march step budget to reduce far-distance misses (holes).
- Moved clipmap page voxel sampling to async worker jobs and limited main-thread page apply budget.
- Added grass-preserving downsample rule for mixed air/surface cells to reduce coarse LOD "inside hill" artifacts.
- Added shader-side transition overlap + stochastic LOD acceptance in a 1-page band to reduce hard LOD pops.
- Fixed stochastic seam weighting direction (coarse now fades out toward inner clipmap center).
- Added LOD-scaled boundary epsilon and shell interval trimming to reduce boundary self-hit artifacts on coarse LODs.
- Removed legacy SVO-DAG render path (`world_ray_march_pipeline`, `world_render`, `svo_upload`) and legacy world shader.
- Simplified world crate to clipmap-focused APIs (`ClipmapStreamingController`, `TerrainGenerator`).
- Removed legacy SVO/DAG voxel modules and exposed clipmap-only voxel APIs.
- Renamed profiler categories/queue metrics to clipmap/page terminology.

## Missing / Not Yet Implemented
- TAA convergence path for stochastic seam blending.
- Motion vectors / per-pixel hit reprojection for TAA.
- Explicit LOD transition band widening logic driven by latency.
- LRU eviction for brick residency.
- occ_l1 / occ_l2 refinement in shader traversal (only occ_l0 used for empty skip).
- GPU layout/validation tests for clipmap structs and interval computation tests.
- Palette overflow stats and pool utilization reporting.

## Known Issues
- LOD seam blending/TAA is still not implemented, so transitions can still pop.
