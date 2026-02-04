Title: Clipmap Brick Pools + Exclusive AABB Shell Traversal

  Status Note (Current)
  This document is the target design plan. Current implementation progress, completed items, and active bugs are tracked in `clipmap_rework_status.md`.
  As of now, async clipmap page generation and shader-side seam softening are implemented, and the legacy SVO-DAG runtime path has been removed.
  TAA/motion vectors and some coarse-LOD artifact fixes are still pending.

  Summary
  This rework replaces SVO‑DAG traversal with a clipmapped sparse voxel grid built from 8³ bricks and dense per‑LOD page
  tables. Rays traverse exclusive AABB shells in a single pass using 0–2 intervals per LOD, sorted front‑to‑back.
  Toroidal addressing keeps page tables stable as the camera moves, while updates only touch newly revealed slices.
  Bricks use separate header and data pools to keep traversal fast (headers accessed without touching large data) and
  avoid fragmentation. LOD transitions use stochastic selection in a blend band (TAA‑friendly). This design maximizes
  cache locality and throughput while supporting instant small edits and batched large updates, and remains compatible
  with all Vulkan 1.3 GPUs (no 64‑bit integer requirement).

  ———

  Key Corrections Applied

  - Brick headers stored in a separate buffer (not embedded in pool entries).
  - Pool strides corrected: palette16 = 288, palette32 = 384, raw16 = 1024.
  - BrickHeader padded explicitly to 32 bytes (8 bytes trailing padding).
  - Palette32 uses 5‑bit packed indices; shader decode is specified.
  - Explicit beyond‑LOD5 fallback: sky/atmosphere.
  - LOD update priority: LOD0 first, coarse LODs later; blend band widened to absorb latency.

  ———

  Public API / Interface Changes

  1. voxelicous-voxel
      - Add ClipmapVoxelStore with:
          - BrickId, PageId, LodLevel, VoxelCoord, WorldCoord
          - Brick encode/decode:
              - palette16, palette32, raw16
          - LOD downsampling:
              - Occupancy: threshold ≥ 2/8 solid children
              - Material: majority vote among solid children (separate from occupancy)
  2. voxelicous-render
      - Use clipmap info via GpuClipmapInfo:
          - Per‑LOD page table base addresses (SoA)
          - Brick header buffer base address
          - Brick pool base addresses (palette16/palette32/raw16)
          - Clipmap origin per LOD (world coords)
          - Voxel size per LOD
          - LOD bounds (AABB min/max per LOD)
      - New compute shader: ray_march_clipmap.comp
  3. voxelicous-world
      - Add ClipmapStreamingController
          - Toroidal indexing for page tables
          - Incremental page invalidation and rebuild
          - Brick residency + LRU

  ———

  Core Data Structures

  BrickHeader Buffer (separate, indexed by BrickId)

  - GPU BrickHeader (32 bytes stride)
      - u8 palette_len
      - u8 encoding (0=pal16, 1=pal32, 2=raw16)
      - u16 flags (bit0: any_transparent, bit1: emissive)
      - u32 data_index (index into the appropriate pool)
      - u32 occ_l0_lo (lower 32 bits of 4×4×4 occupancy)
      - u32 occ_l0_hi (upper 32 bits)
      - u8 occ_l1 (2×2×2 occupancy)
      - u8 occ_l2 (1×1×1 occupancy or derive from occ_l1)
      - u16 padding
      - u32 avg_color (optional debug/preview)
      - u32 _pad0 (explicit)
      - u32 _pad1 (explicit)

  Brick Pools (data only)

  - palette16_pool: 288 bytes per entry
      - palette: 16 × u16 = 32 bytes
      - indices: 512 × 4 bits = 256 bytes
  - palette32_pool: 384 bytes per entry
      - palette: 32 × u16 = 64 bytes
      - indices: 512 × 5 bits = 320 bytes
  - raw16_pool: 1024 bytes per entry
      - data: 512 × u16

  Palette32 Decode (GLSL)

  uint bit_idx  = voxel_idx * 5u;
  uint byte_idx = bit_idx >> 3u;
  uint bit_off  = bit_idx & 7u;
  uint raw = data[byte_idx] | (data[byte_idx + 1u] << 8u);
  uint mat_idx = (raw >> bit_off) & 0x1Fu;

  This ALU cost is accepted to save bandwidth; palette32 reduces pressure on raw16.

  ———

  Page Tables (per LOD, SoA layout)

  - Grid size: 16×16×16 = 4096 pages per LOD
  - page_brick_indices[] — u32[64] per page, padded to 256 bytes
  - page_occ[] — uvec2 (64‑bit occupancy via 2×u32)

  ———

  Toroidal Addressing

  - Page index per axis:
      - page_ix = floor(world_voxel / page_size) % page_grid
  - When clipmap origin shifts:
      - Invalidate only the newly revealed slice (at most 16×16 pages).
      - Existing pages retain their indices; no full shifts.

  ———

  LOD Coverage

  - 6 LODs, voxel size doubles each LOD:
      - LOD0: size 1, coverage 512 (radius ~256)
      - LOD1: size 2, coverage 1024 (radius ~512)
      - LOD2: size 4, coverage 2048 (radius ~1024)
      - LOD3: size 8, coverage 4096 (radius ~2048)
      - LOD4: size 16, coverage 8192 (radius ~4096)
      - LOD5: size 32, coverage 16384 (radius ~8192)

  ———

  Exclusive LOD Shell Traversal (AABB, Single Pass)

  1. For each LOD, compute t_enter/t_exit with its AABB.
  2. For LODk (k>0), compute exclusive region = LODk_AABB minus LOD(k−1)_AABB.
      - This can yield 0, 1, or 2 disjoint intervals.
  3. Build interval list:
      - Up to 12 intervals (6 LOD × 2).
  4. Sort intervals by t_near and traverse front‑to‑back.
  5. Early exit on first hit.

  ———

  Traversal Algorithm (per interval)

  1. DDA through pages along the interval range.
  2. Skip empty pages using page_occ (uvec2).
  3. DDA through bricks (4×4×4 per page).
  4. Skip empty bricks using BrickHeader.occ_l0.
  5. Refine with occ_l1, then decode voxel.

  ———

  LOD Transition Seams

  - Stochastic LOD selection in a transition band:
      - Randomly choose LOD based on distance within band.
      - Requires TAA for convergence.
  - Blend band widened by 1 page to absorb 1–2 frame lag between LOD updates.

  ———

  Downsampling Strategy

  - Occupancy decision: parent solid if ≥2 of 8 children are solid.
  - Material decision: majority vote among solid children (separate from occupancy).
  - Tie‑breaker: lowest material ID.

  ———

  Normals

  - Face normals from hit voxel face (cheap, stable across LODs).

  ———

  Motion Vectors for TAA

  - Store per‑pixel hit world position in a buffer.
  - Reproject with previous view/projection to generate motion vectors.

  ———

  LOD Update Priority

  - Update LOD0 first every frame when camera moves.
  - Coarser LODs update in subsequent frames (amortized).
  - Transition band widened to hide stale‑data blending.

  ———

  Beyond‑LOD5 Fallback

  - If no hit in any interval, output sky/atmosphere (default: gradient skybox).
  - This is the defined traversal termination condition.

  ———

  CPU↔GPU Sync

  - Persistent staging ring buffer.
  - Small edits: update affected brick + page immediately.
  - Large edits: batch into regions, stream over multiple frames.
  - LRU eviction for bricks not referenced by any page.

  ———

  Testing & Validation

  1. Brick encoding round‑trip (pal16/pal32/raw16).
  2. Occupancy masks correctness (occ_l0, occ_l1, occ_l2).
  3. Toroidal indexing + slice invalidation.
  4. Exclusive interval computation (0–2 intervals per LOD).
  5. LOD seam noise convergence with TAA.
  6. GPU struct layout checks (Rust ↔ GLSL).
  7. Pool utilization & palette overflow stats.

  ———

  Assumptions & Defaults

  - LOD count = 6, page grid = 16³, brick size = 8³.
  - Palette tiers: 16 + 32 + raw16.
  - Headers stored separately (32‑byte stride).
  - AABB bounds for clipmap volumes.
  - Stochastic seam handling on by default.
  - All‑air bricks never allocated (empty slots = index 0).
  - Portable occupancy masks (uvec2, no u64).
