//! Clipmap streaming controller for brick/page management.

use std::{
    collections::{HashMap, VecDeque},
    sync::mpsc::{self, Receiver, Sender, TryRecvError},
    sync::Arc,
};

use glam::Vec3;
use voxelicous_core::types::BlockId;
use voxelicous_voxel::{
    downsample_voxel, BrickEncoding, BrickId, ClipmapVoxelStore, WorldCoord, BRICK_SIZE,
    BRICK_VOXELS, CLIPMAP_LOD_COUNT, CLIPMAP_PAGE_GRID, PAGE_BRICKS, PAGE_BRICKS_PER_AXIS,
    PAGE_VOXELS_PER_AXIS,
};

use crate::generation::TerrainGenerator;

/// Dirty ranges to upload to GPU after a clipmap update.
#[derive(Debug, Default)]
pub struct ClipmapDirtyState {
    pub dirty_pages: Vec<Vec<usize>>,
    pub dirty_headers: Vec<BrickId>,
    pub dirty_palette16_entries: Vec<u32>,
    pub dirty_palette32_entries: Vec<u32>,
    pub dirty_raw16_entries: Vec<u32>,
}

#[derive(Clone, Debug)]
struct ClipmapLodState {
    origin: Option<WorldCoord>,
    page_brick_indices: Vec<u32>,
    page_occ: Vec<[u32; 2]>,
    page_coords: Vec<[i32; 4]>,
    page_loaded: Vec<bool>,
    loaded_pages: usize,
    dirty_pages: Vec<usize>,
    pending_pages: VecDeque<(i64, i64, i64)>,
    generation: u64,
    inflight_pages: usize,
    ready: bool,
}

impl ClipmapLodState {
    fn new() -> Self {
        let page_count = CLIPMAP_PAGE_GRID * CLIPMAP_PAGE_GRID * CLIPMAP_PAGE_GRID;
        Self {
            origin: None,
            page_brick_indices: vec![0; page_count * PAGE_BRICKS],
            page_occ: vec![[0, 0]; page_count],
            page_coords: vec![invalid_page_coord(); page_count],
            page_loaded: vec![false; page_count],
            loaded_pages: 0,
            dirty_pages: Vec::new(),
            pending_pages: VecDeque::new(),
            generation: 0,
            inflight_pages: 0,
            ready: false,
        }
    }
}

#[derive(Clone, Debug)]
struct BuiltPage {
    coord: (i64, i64, i64),
    bricks: Vec<[BlockId; BRICK_VOXELS]>,
    occ: u64,
}

#[derive(Clone, Debug)]
struct PageBuildResult {
    lod: usize,
    generation: u64,
    page: BuiltPage,
}

/// Clipmap streaming controller (toroidal page tables + brick pools).
pub struct ClipmapStreamingController {
    generator: TerrainGenerator,
    edits: HashMap<WorldCoord, BlockId>,
    edit_snapshot: Arc<HashMap<WorldCoord, BlockId>>,
    store: ClipmapVoxelStore,
    lods: Vec<ClipmapLodState>,
    camera_voxel: WorldCoord,
    frame_counter: u64,
    coarse_lod_cursor: usize,
    bootstrap_lod: usize,
    dirty_headers: Vec<BrickId>,
    dirty_palette16_entries: Vec<u32>,
    dirty_palette32_entries: Vec<u32>,
    dirty_raw16_entries: Vec<u32>,
    page_build_tx: Sender<PageBuildResult>,
    page_build_rx: Receiver<PageBuildResult>,
    inflight_jobs: usize,
    pending_brick_frees: VecDeque<(u64, BrickId)>,
}

impl ClipmapStreamingController {
    const PAGE_APPLY_BUDGET_STEADY: usize = 2;
    const PAGE_APPLY_BUDGET_BOOTSTRAP: usize = 12;
    const MAX_INFLIGHT_PAGE_JOBS: usize = 16;
    const BRICK_FREE_DELAY_FRAMES: u64 = 3;
    const SYNC_EDIT_LODS: usize = 2;

    /// Create a new clipmap streaming controller.
    pub fn new(generator: TerrainGenerator) -> Self {
        let (page_build_tx, page_build_rx) = mpsc::channel();
        let lods = (0..CLIPMAP_LOD_COUNT)
            .map(|_| ClipmapLodState::new())
            .collect();
        Self {
            generator,
            edits: HashMap::new(),
            edit_snapshot: Arc::new(HashMap::new()),
            store: ClipmapVoxelStore::new(),
            lods,
            camera_voxel: WorldCoord { x: 0, y: 0, z: 0 },
            frame_counter: 0,
            coarse_lod_cursor: 1,
            bootstrap_lod: 0,
            dirty_headers: Vec::new(),
            dirty_palette16_entries: Vec::new(),
            dirty_palette32_entries: Vec::new(),
            dirty_raw16_entries: Vec::new(),
            page_build_tx,
            page_build_rx,
            inflight_jobs: 0,
            pending_brick_frees: VecDeque::new(),
        }
    }

    /// Sample block id at world voxel coordinates, including runtime edits.
    pub fn block_at_world(&self, x: i64, y: i64, z: i64) -> BlockId {
        let coord = WorldCoord { x, y, z };
        self.edits
            .get(&coord)
            .copied()
            .unwrap_or_else(|| self.generator.block_at_world(x, y, z))
    }

    /// Set a block id at world voxel coordinates.
    ///
    /// Returns `true` when the effective block value changed.
    pub fn set_block_at_world(&mut self, x: i64, y: i64, z: i64, block: BlockId) -> bool {
        let coord = WorldCoord { x, y, z };
        let previous = self.block_at_world(x, y, z);
        if previous == block {
            return false;
        }

        // Store only differences from procedural terrain.
        let generated = self.generator.block_at_world(x, y, z);
        if block == generated {
            self.edits.remove(&coord);
        } else {
            self.edits.insert(coord, block);
        }
        self.edit_snapshot = Arc::new(self.edits.clone());

        self.apply_edit_immediate(coord);
        self.enqueue_pages_affected_by_edit(coord);
        true
    }

    /// Destroy (set to air) the block at world voxel coordinates.
    ///
    /// Returns `true` when a solid block was destroyed.
    pub fn destroy_block_at_world(&mut self, x: i64, y: i64, z: i64) -> bool {
        if self.block_at_world(x, y, z).is_air() {
            return false;
        }
        self.set_block_at_world(x, y, z, BlockId::AIR)
    }

    /// Update the clipmap around the given camera position (world units).
    #[cfg_attr(
        feature = "profiling-tracy",
        tracing::instrument(level = "trace", skip_all)
    )]
    pub fn update(&mut self, camera_pos: Vec3) {
        self.process_deferred_brick_frees();

        let camera_voxel = WorldCoord {
            x: camera_pos.x.floor() as i64,
            y: camera_pos.y.floor() as i64,
            z: camera_pos.z.floor() as i64,
        };
        self.camera_voxel = camera_voxel;
        // First update: seed origins and enqueue LOD0 only to avoid long stalls.
        if self.lods.iter().any(|lod| lod.origin.is_none()) {
            for lod in 0..CLIPMAP_LOD_COUNT {
                let voxel_size = self.lod_voxel_size(lod);
                let page_size = PAGE_VOXELS_PER_AXIS as i64 * voxel_size;
                let coverage = self.lod_coverage(lod);
                let origin = aligned_origin(camera_voxel, coverage, page_size);
                self.lods[lod].origin = Some(origin);
            }

            let voxel_size0 = self.lod_voxel_size(0);
            let page_size0 = PAGE_VOXELS_PER_AXIS as i64 * voxel_size0;
            let coverage0 = self.lod_coverage(0);
            let origin0 = aligned_origin(camera_voxel, coverage0, page_size0);
            self.enqueue_full_rebuild(0, origin0, voxel_size0, page_size0);
            self.bootstrap_lod = 0;
        } else if self.bootstrap_lod < CLIPMAP_LOD_COUNT {
            // While bootstrapping, keep LOD0 updated and enqueue next LOD once current is filled.
            self.update_lod(0, camera_voxel, false);

            for lod in 1..self.bootstrap_lod {
                self.update_lod(lod, camera_voxel, false);
            }

            if self.lods[self.bootstrap_lod].pending_pages.is_empty() {
                self.bootstrap_lod += 1;
                if self.bootstrap_lod < CLIPMAP_LOD_COUNT {
                    let voxel_size = self.lod_voxel_size(self.bootstrap_lod);
                    let page_size = PAGE_VOXELS_PER_AXIS as i64 * voxel_size;
                    let coverage = self.lod_coverage(self.bootstrap_lod);
                    let origin = aligned_origin(camera_voxel, coverage, page_size);
                    self.lods[self.bootstrap_lod].origin = Some(origin);
                    self.enqueue_full_rebuild(self.bootstrap_lod, origin, voxel_size, page_size);
                }
            }
        } else {
            // Always update LOD0.
            self.update_lod(0, camera_voxel, false);

            // Stagger coarse LOD updates.
            let lod = self.coarse_lod_cursor;
            self.update_lod(lod, camera_voxel, false);
            self.coarse_lod_cursor += 1;
            if self.coarse_lod_cursor >= CLIPMAP_LOD_COUNT {
                self.coarse_lod_cursor = 1;
            }
        }

        let apply_budget = if self.bootstrap_lod < CLIPMAP_LOD_COUNT {
            Self::PAGE_APPLY_BUDGET_BOOTSTRAP
        } else {
            Self::PAGE_APPLY_BUDGET_STEADY
        };
        self.process_pending_pages(apply_budget);
        self.frame_counter = self.frame_counter.wrapping_add(1);
    }

    /// Take and clear the dirty state accumulated during updates.
    pub fn take_dirty_state(&mut self) -> ClipmapDirtyState {
        let dirty_pages = self
            .lods
            .iter_mut()
            .map(|lod| std::mem::take(&mut lod.dirty_pages))
            .collect();

        ClipmapDirtyState {
            dirty_pages,
            dirty_headers: std::mem::take(&mut self.dirty_headers),
            dirty_palette16_entries: std::mem::take(&mut self.dirty_palette16_entries),
            dirty_palette32_entries: std::mem::take(&mut self.dirty_palette32_entries),
            dirty_raw16_entries: std::mem::take(&mut self.dirty_raw16_entries),
        }
    }

    /// Access the clipmap voxel store (for GPU upload).
    pub fn store(&self) -> &ClipmapVoxelStore {
        &self.store
    }

    /// Access the clipmap voxel store mutably (for tooling).
    pub fn store_mut(&mut self) -> &mut ClipmapVoxelStore {
        &mut self.store
    }

    /// Get page brick indices for a given LOD (SoA).
    pub fn page_brick_indices(&self, lod: usize) -> &[u32] {
        &self.lods[lod].page_brick_indices
    }

    /// Get page occupancy for a given LOD.
    pub fn page_occ(&self, lod: usize) -> &[[u32; 2]] {
        &self.lods[lod].page_occ
    }

    /// Get owning world page coordinates for a given LOD slot.
    pub fn page_coords(&self, lod: usize) -> &[[i32; 4]] {
        &self.lods[lod].page_coords
    }

    /// Get clipmap origin (min corner) for a given LOD.
    pub fn lod_origin(&self, lod: usize) -> WorldCoord {
        self.lods[lod]
            .origin
            .unwrap_or(WorldCoord { x: 0, y: 0, z: 0 })
    }

    /// Get voxel size for a given LOD in base voxels.
    pub fn lod_voxel_size(&self, lod: usize) -> i64 {
        1i64 << lod
    }

    /// Get coverage (extent) for a given LOD in base voxels.
    pub fn lod_coverage(&self, lod: usize) -> i64 {
        let voxels = (CLIPMAP_PAGE_GRID * PAGE_VOXELS_PER_AXIS) as i64;
        voxels * self.lod_voxel_size(lod)
    }

    /// Returns true if this LOD has completed at least one full build.
    pub fn lod_ready(&self, lod: usize) -> bool {
        self.lods[lod].ready
    }

    /// Returns true if this LOD has at least one loaded page.
    pub fn lod_renderable(&self, lod: usize) -> bool {
        self.lods[lod].loaded_pages > 0
    }

    #[cfg_attr(
        feature = "profiling-tracy",
        tracing::instrument(level = "trace", skip_all)
    )]
    fn update_lod(&mut self, lod: usize, camera_voxel: WorldCoord, force: bool) {
        let voxel_size = self.lod_voxel_size(lod);
        let page_size = PAGE_VOXELS_PER_AXIS as i64 * voxel_size;
        let coverage = self.lod_coverage(lod);
        let origin = aligned_origin(camera_voxel, coverage, page_size);

        let old_origin = self.lods[lod].origin.unwrap_or(origin);
        let shift = if force {
            (
                CLIPMAP_PAGE_GRID as i64,
                CLIPMAP_PAGE_GRID as i64,
                CLIPMAP_PAGE_GRID as i64,
            )
        } else {
            (
                (origin.x - old_origin.x) / page_size,
                (origin.y - old_origin.y) / page_size,
                (origin.z - old_origin.z) / page_size,
            )
        };

        let max_shift = shift.0.abs().max(shift.1.abs()).max(shift.2.abs());

        if force || max_shift as usize >= CLIPMAP_PAGE_GRID {
            self.enqueue_full_rebuild(lod, origin, voxel_size, page_size);
            return;
        }

        if shift.0 == 0 && shift.1 == 0 && shift.2 == 0 {
            self.lods[lod].origin = Some(origin);
            return;
        }

        let page_origin = (
            div_floor(origin.x, page_size),
            div_floor(origin.y, page_size),
            div_floor(origin.z, page_size),
        );

        if shift.0 != 0 {
            self.enqueue_slice(lod, page_origin, Axis::X, shift.0);
        }
        if shift.1 != 0 {
            self.enqueue_slice(lod, page_origin, Axis::Y, shift.1);
        }
        if shift.2 != 0 {
            self.enqueue_slice(lod, page_origin, Axis::Z, shift.2);
        }

        self.lods[lod].origin = Some(origin);
    }

    fn enqueue_full_rebuild(
        &mut self,
        lod: usize,
        origin: WorldCoord,
        _voxel_size: i64,
        page_size: i64,
    ) {
        // Release bricks currently referenced by this LOD before wiping page tables.
        let mut old_ids = Vec::new();
        for raw_id in self.lods[lod].page_brick_indices.iter().copied() {
            if raw_id != 0 {
                old_ids.push(BrickId(raw_id));
            }
        }
        for id in old_ids {
            self.queue_free_brick(id);
        }

        let page_origin = (
            div_floor(origin.x, page_size),
            div_floor(origin.y, page_size),
            div_floor(origin.z, page_size),
        );

        let page_count = CLIPMAP_PAGE_GRID * CLIPMAP_PAGE_GRID * CLIPMAP_PAGE_GRID;
        {
            let lod_state = &mut self.lods[lod];
            lod_state.generation = lod_state.generation.wrapping_add(1);
            lod_state.origin = Some(origin);
            lod_state.pending_pages.clear();
            lod_state.pending_pages.reserve(page_count);
            lod_state.page_brick_indices.as_mut_slice().fill(0);
            for occ in &mut lod_state.page_occ {
                *occ = [0, 0];
            }
            lod_state
                .page_coords
                .as_mut_slice()
                .fill(invalid_page_coord());
            lod_state.page_loaded.as_mut_slice().fill(false);
            lod_state.loaded_pages = 0;
            lod_state.dirty_pages.clear();
            lod_state.dirty_pages.extend(0..page_count);
            lod_state.ready = false;
        }

        let mut coords = Vec::with_capacity(page_count);
        for z in 0..CLIPMAP_PAGE_GRID {
            for y in 0..CLIPMAP_PAGE_GRID {
                for x in 0..CLIPMAP_PAGE_GRID {
                    coords.push((
                        page_origin.0 + x as i64,
                        page_origin.1 + y as i64,
                        page_origin.2 + z as i64,
                    ));
                }
            }
        }

        let camera_voxel = self.camera_voxel;
        coords.sort_unstable_by_key(|&coord| {
            page_distance_to_camera_sq(coord, camera_voxel, page_size)
        });
        self.lods[lod].pending_pages.extend(coords);
    }

    fn enqueue_slice(&mut self, lod: usize, page_origin: (i64, i64, i64), axis: Axis, shift: i64) {
        let count = shift.unsigned_abs() as usize;
        let grid = CLIPMAP_PAGE_GRID as i64;

        let (start, end) = if shift > 0 {
            (grid - count as i64, grid)
        } else {
            (0, count as i64)
        };

        let mut coords = Vec::with_capacity(count * CLIPMAP_PAGE_GRID * CLIPMAP_PAGE_GRID);
        for idx in start..end {
            for j in 0..CLIPMAP_PAGE_GRID as i64 {
                for k in 0..CLIPMAP_PAGE_GRID as i64 {
                    let (px, py, pz) = match axis {
                        Axis::X => (page_origin.0 + idx, page_origin.1 + j, page_origin.2 + k),
                        Axis::Y => (page_origin.0 + j, page_origin.1 + idx, page_origin.2 + k),
                        Axis::Z => (page_origin.0 + j, page_origin.1 + k, page_origin.2 + idx),
                    };
                    coords.push((px, py, pz));
                }
            }
        }

        // Clear incoming toroidal slots immediately so stale pages do not appear at new locations.
        for &coord in &coords {
            self.invalidate_page_slot(lod, coord);
        }

        let page_size = PAGE_VOXELS_PER_AXIS as i64 * self.lod_voxel_size(lod);
        let camera_voxel = self.camera_voxel;
        coords.sort_unstable_by_key(|&coord| {
            page_distance_to_camera_sq(coord, camera_voxel, page_size)
        });
        self.lods[lod].pending_pages.extend(coords);
    }

    #[cfg_attr(
        feature = "profiling-tracy",
        tracing::instrument(level = "trace", skip_all)
    )]
    fn process_pending_pages(&mut self, mut apply_budget: usize) {
        self.spawn_pending_jobs();

        while apply_budget > 0 {
            let result = match self.page_build_rx.try_recv() {
                Ok(result) => result,
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            };

            self.inflight_jobs = self.inflight_jobs.saturating_sub(1);
            let lod_state = &mut self.lods[result.lod];
            lod_state.inflight_pages = lod_state.inflight_pages.saturating_sub(1);

            if result.generation != lod_state.generation {
                continue;
            }

            self.apply_built_page(result.lod, result.page);
            apply_budget -= 1;
        }

        for lod in 0..CLIPMAP_LOD_COUNT {
            let state = &mut self.lods[lod];
            if state.origin.is_some() && state.pending_pages.is_empty() && state.inflight_pages == 0
            {
                state.ready = true;
            }
        }
    }

    fn spawn_pending_jobs(&mut self) {
        while self.inflight_jobs < Self::MAX_INFLIGHT_PAGE_JOBS {
            let Some((lod, coord, voxel_size, generation)) = self.pop_next_pending_page() else {
                break;
            };

            self.inflight_jobs += 1;
            self.lods[lod].inflight_pages += 1;

            let tx = self.page_build_tx.clone();
            let generator = self.generator.clone();
            let edits = Arc::clone(&self.edit_snapshot);
            rayon::spawn(move || {
                let page = build_page_voxels(&generator, &edits, coord, voxel_size);
                let _ = tx.send(PageBuildResult {
                    lod,
                    generation,
                    page,
                });
            });
        }
    }

    fn pop_next_pending_page(&mut self) -> Option<(usize, (i64, i64, i64), i64, u64)> {
        for lod in 0..CLIPMAP_LOD_COUNT {
            let voxel_size = self.lod_voxel_size(lod);
            let generation = self.lods[lod].generation;
            while let Some(coord) = self.lods[lod].pending_pages.pop_front() {
                if !self.is_page_in_coverage(lod, coord) {
                    continue;
                }
                return Some((lod, coord, voxel_size, generation));
            }
        }
        None
    }

    fn apply_built_page(&mut self, lod: usize, page: BuiltPage) {
        let page_coord = page.coord;
        if !self.is_page_in_coverage(lod, page_coord) {
            return;
        }

        let page_index = Self::page_index_from_coord(page_coord);

        self.clear_page_slot(lod, page_index);

        let base_offset = page_index * PAGE_BRICKS;
        let mut occ: u64 = 0;
        for (brick_idx, voxels) in page.bricks.iter().enumerate() {
            let brick_id = self.store.allocate_brick(voxels);
            self.lods[lod].page_brick_indices[base_offset + brick_idx] = brick_id.0;

            if brick_id.0 != 0 {
                occ |= 1u64 << brick_idx;
                self.mark_brick_dirty(brick_id);
            }
        }

        let final_occ = if page.occ == 0 { occ } else { page.occ };
        self.lods[lod].page_occ[page_index] =
            [(final_occ & 0xFFFF_FFFF) as u32, (final_occ >> 32) as u32];
        self.lods[lod].page_coords[page_index] = [
            page_coord.0 as i32,
            page_coord.1 as i32,
            page_coord.2 as i32,
            0,
        ];
        if !self.lods[lod].page_loaded[page_index] {
            self.lods[lod].page_loaded[page_index] = true;
            self.lods[lod].loaded_pages += 1;
        }
        self.lods[lod].dirty_pages.push(page_index);
    }

    fn invalidate_page_slot(&mut self, lod: usize, page_coord: (i64, i64, i64)) {
        let page_index = Self::page_index_from_coord(page_coord);
        self.clear_page_slot(lod, page_index);
    }

    fn clear_page_slot(&mut self, lod: usize, page_index: usize) {
        let base_offset = page_index * PAGE_BRICKS;
        let mut had_data = false;
        let mut ids_to_free = Vec::new();
        for i in 0..PAGE_BRICKS {
            let slot = base_offset + i;
            let id = BrickId(self.lods[lod].page_brick_indices[slot]);
            if id.0 != 0 {
                ids_to_free.push(id);
                had_data = true;
            }
            self.lods[lod].page_brick_indices[slot] = 0;
        }

        if self.lods[lod].page_occ[page_index] != [0, 0] {
            had_data = true;
        }
        self.lods[lod].page_occ[page_index] = [0, 0];
        if self.lods[lod].page_coords[page_index] != invalid_page_coord() {
            self.lods[lod].page_coords[page_index] = invalid_page_coord();
            had_data = true;
        }
        if self.lods[lod].page_loaded[page_index] {
            self.lods[lod].page_loaded[page_index] = false;
            self.lods[lod].loaded_pages = self.lods[lod].loaded_pages.saturating_sub(1);
            had_data = true;
        }

        if had_data {
            self.lods[lod].dirty_pages.push(page_index);
        }

        for id in ids_to_free {
            self.queue_free_brick(id);
        }
    }

    fn queue_free_brick(&mut self, id: BrickId) {
        if id.0 == 0 {
            return;
        }
        let release_frame = self
            .frame_counter
            .wrapping_add(Self::BRICK_FREE_DELAY_FRAMES);
        self.pending_brick_frees.push_back((release_frame, id));
    }

    fn process_deferred_brick_frees(&mut self) {
        while let Some((release_frame, id)) = self.pending_brick_frees.front().copied() {
            if release_frame > self.frame_counter {
                break;
            }
            self.pending_brick_frees.pop_front();
            self.store.free_brick(id);
        }
    }

    fn page_index_from_coord(page_coord: (i64, i64, i64)) -> usize {
        let page_ix = mod_floor(page_coord.0, CLIPMAP_PAGE_GRID as i64) as usize;
        let page_iy = mod_floor(page_coord.1, CLIPMAP_PAGE_GRID as i64) as usize;
        let page_iz = mod_floor(page_coord.2, CLIPMAP_PAGE_GRID as i64) as usize;
        page_ix + page_iy * CLIPMAP_PAGE_GRID + page_iz * CLIPMAP_PAGE_GRID * CLIPMAP_PAGE_GRID
    }

    fn is_page_in_coverage(&self, lod: usize, page_coord: (i64, i64, i64)) -> bool {
        let Some(origin) = self.lods[lod].origin else {
            return false;
        };
        let page_size = PAGE_VOXELS_PER_AXIS as i64 * self.lod_voxel_size(lod);
        let origin_page = (
            div_floor(origin.x, page_size),
            div_floor(origin.y, page_size),
            div_floor(origin.z, page_size),
        );
        let grid = CLIPMAP_PAGE_GRID as i64;

        page_coord.0 >= origin_page.0
            && page_coord.0 < origin_page.0 + grid
            && page_coord.1 >= origin_page.1
            && page_coord.1 < origin_page.1 + grid
            && page_coord.2 >= origin_page.2
            && page_coord.2 < origin_page.2 + grid
    }

    fn mark_brick_dirty(&mut self, brick_id: BrickId) {
        if let Some(header) = self.store.header(brick_id) {
            self.dirty_headers.push(brick_id);
            match BrickEncoding::from_u8(header.encoding) {
                Some(BrickEncoding::Palette16) => {
                    self.dirty_palette16_entries.push(header.data_index);
                }
                Some(BrickEncoding::Palette32) => {
                    self.dirty_palette32_entries.push(header.data_index);
                }
                Some(BrickEncoding::Raw16) => {
                    self.dirty_raw16_entries.push(header.data_index);
                }
                None => {}
            }
        }
    }

    fn apply_edit_immediate(&mut self, world: WorldCoord) {
        let sync_lods = Self::SYNC_EDIT_LODS.min(CLIPMAP_LOD_COUNT);
        let edits_snapshot = Arc::clone(&self.edit_snapshot);

        for lod in 0..sync_lods {
            let affected_pages = self.affected_pages_for_edit(lod, world);
            let voxel_size = self.lod_voxel_size(lod);
            for page_coord in affected_pages {
                if !self.is_page_in_coverage(lod, page_coord) {
                    continue;
                }

                let page =
                    build_page_voxels(&self.generator, &edits_snapshot, page_coord, voxel_size);
                self.apply_built_page(lod, page);
                self.lods[lod]
                    .pending_pages
                    .retain(|&coord| coord != page_coord);
            }
            self.lods[lod].ready = false;
        }
    }

    fn enqueue_pages_affected_by_edit(&mut self, world: WorldCoord) {
        for lod in Self::SYNC_EDIT_LODS.min(CLIPMAP_LOD_COUNT)..CLIPMAP_LOD_COUNT {
            if self.lods[lod].origin.is_none() {
                continue;
            }

            for page_coord in self.affected_pages_for_edit(lod, world) {
                if !self.is_page_in_coverage(lod, page_coord) {
                    continue;
                }
                if !self.lods[lod].pending_pages.contains(&page_coord) {
                    self.lods[lod].pending_pages.push_front(page_coord);
                }
                self.lods[lod].ready = false;
            }
        }
    }

    fn affected_pages_for_edit(&self, lod: usize, world: WorldCoord) -> Vec<(i64, i64, i64)> {
        let voxel_size = self.lod_voxel_size(lod);
        let half = voxel_size / 2;
        let mut xs = vec![world.x];
        let mut ys = vec![world.y];
        let mut zs = vec![world.z];
        if half > 0 {
            xs.push(world.x - half);
            ys.push(world.y - half);
            zs.push(world.z - half);
        }

        let mut affected_pages = Vec::with_capacity(8);
        let page_size = PAGE_VOXELS_PER_AXIS as i64 * voxel_size;
        for &x in &xs {
            for &y in &ys {
                for &z in &zs {
                    let sample_origin = (
                        div_floor(x, voxel_size) * voxel_size,
                        div_floor(y, voxel_size) * voxel_size,
                        div_floor(z, voxel_size) * voxel_size,
                    );
                    let page_coord = (
                        div_floor(sample_origin.0, page_size),
                        div_floor(sample_origin.1, page_size),
                        div_floor(sample_origin.2, page_size),
                    );
                    if !affected_pages.contains(&page_coord) {
                        affected_pages.push(page_coord);
                    }
                }
            }
        }

        affected_pages
    }
}

#[derive(Clone, Copy)]
enum Axis {
    X,
    Y,
    Z,
}

fn aligned_origin(camera: WorldCoord, coverage: i64, page_size: i64) -> WorldCoord {
    let half = coverage / 2;
    let ox = div_floor(camera.x - half, page_size) * page_size;
    let oy = div_floor(camera.y - half, page_size) * page_size;
    let oz = div_floor(camera.z - half, page_size) * page_size;
    WorldCoord {
        x: ox,
        y: oy,
        z: oz,
    }
}

fn page_distance_to_camera_sq(
    page_coord: (i64, i64, i64),
    camera_voxel: WorldCoord,
    page_size: i64,
) -> i128 {
    let half_page = page_size / 2;
    let center_x = i128::from(page_coord.0) * i128::from(page_size) + i128::from(half_page);
    let center_y = i128::from(page_coord.1) * i128::from(page_size) + i128::from(half_page);
    let center_z = i128::from(page_coord.2) * i128::from(page_size) + i128::from(half_page);

    let dx = center_x - i128::from(camera_voxel.x);
    let dy = center_y - i128::from(camera_voxel.y);
    let dz = center_z - i128::from(camera_voxel.z);

    dx * dx + dy * dy + dz * dz
}

#[cfg_attr(
    feature = "profiling-tracy",
    tracing::instrument(level = "trace", skip_all)
)]
fn build_page_voxels(
    generator: &TerrainGenerator,
    edits: &HashMap<WorldCoord, BlockId>,
    page_coord: (i64, i64, i64),
    voxel_size: i64,
) -> BuiltPage {
    let page_size = PAGE_VOXELS_PER_AXIS as i64 * voxel_size;
    let page_origin = WorldCoord {
        x: page_coord.0 * page_size,
        y: page_coord.1 * page_size,
        z: page_coord.2 * page_size,
    };

    if voxel_size == 1 {
        return build_page_voxels_unit_lod(generator, edits, page_coord, page_origin);
    }

    let mut occ: u64 = 0;
    let mut bricks = Vec::with_capacity(PAGE_BRICKS);

    for bz in 0..PAGE_BRICKS_PER_AXIS {
        for by in 0..PAGE_BRICKS_PER_AXIS {
            for bx in 0..PAGE_BRICKS_PER_AXIS {
                let brick_origin = WorldCoord {
                    x: page_origin.x + (bx * BRICK_SIZE) as i64 * voxel_size,
                    y: page_origin.y + (by * BRICK_SIZE) as i64 * voxel_size,
                    z: page_origin.z + (bz * BRICK_SIZE) as i64 * voxel_size,
                };

                let mut voxels = [BlockId::AIR; BRICK_VOXELS];
                let mut any_solid = false;
                for z in 0..BRICK_SIZE {
                    for y in 0..BRICK_SIZE {
                        for x in 0..BRICK_SIZE {
                            let world_x = brick_origin.x + (x as i64) * voxel_size;
                            let world_y = brick_origin.y + (y as i64) * voxel_size;
                            let world_z = brick_origin.z + (z as i64) * voxel_size;

                            let idx = x + y * BRICK_SIZE + z * BRICK_SIZE * BRICK_SIZE;
                            let block = sample_voxel_from_generator(
                                generator, edits, world_x, world_y, world_z, voxel_size,
                            );
                            voxels[idx] = block;
                            any_solid |= block.is_solid();
                        }
                    }
                }

                let brick_idx = bx
                    + by * PAGE_BRICKS_PER_AXIS
                    + bz * PAGE_BRICKS_PER_AXIS * PAGE_BRICKS_PER_AXIS;
                if any_solid {
                    occ |= 1u64 << brick_idx;
                }
                bricks.push(voxels);
            }
        }
    }

    BuiltPage {
        coord: page_coord,
        bricks,
        occ,
    }
}

#[cfg_attr(
    feature = "profiling-tracy",
    tracing::instrument(level = "trace", skip_all)
)]
fn build_page_voxels_unit_lod(
    generator: &TerrainGenerator,
    edits: &HashMap<WorldCoord, BlockId>,
    page_coord: (i64, i64, i64),
    page_origin: WorldCoord,
) -> BuiltPage {
    let mut occ: u64 = 0;
    let mut bricks = Vec::with_capacity(PAGE_BRICKS);
    let mut surface_heights = vec![0i32; PAGE_VOXELS_PER_AXIS * PAGE_VOXELS_PER_AXIS];
    let mut surface_blocks = vec![BlockId::AIR; PAGE_VOXELS_PER_AXIS * PAGE_VOXELS_PER_AXIS];

    for z in 0..PAGE_VOXELS_PER_AXIS {
        for x in 0..PAGE_VOXELS_PER_AXIS {
            let world_x = page_origin.x + x as i64;
            let world_z = page_origin.z + z as i64;
            let sample = generator.surface_at(world_x, world_z);
            let index = x + z * PAGE_VOXELS_PER_AXIS;
            surface_heights[index] = sample.surface_height;
            surface_blocks[index] = sample.top_block;
        }
    }

    let dirt_depth = i64::from(generator.config().dirt_depth);
    for bz in 0..PAGE_BRICKS_PER_AXIS {
        for by in 0..PAGE_BRICKS_PER_AXIS {
            for bx in 0..PAGE_BRICKS_PER_AXIS {
                let brick_origin = WorldCoord {
                    x: page_origin.x + (bx * BRICK_SIZE) as i64,
                    y: page_origin.y + (by * BRICK_SIZE) as i64,
                    z: page_origin.z + (bz * BRICK_SIZE) as i64,
                };

                let mut voxels = [BlockId::AIR; BRICK_VOXELS];
                let mut any_solid = false;
                for z in 0..BRICK_SIZE {
                    for y in 0..BRICK_SIZE {
                        for x in 0..BRICK_SIZE {
                            let world_x = brick_origin.x + x as i64;
                            let world_y = brick_origin.y + y as i64;
                            let world_z = brick_origin.z + z as i64;
                            let page_x = bx * BRICK_SIZE + x;
                            let page_z = bz * BRICK_SIZE + z;
                            let index = page_x + page_z * PAGE_VOXELS_PER_AXIS;
                            let surface_height = i64::from(surface_heights[index]);
                            let surface_block = surface_blocks[index];
                            let idx = x + y * BRICK_SIZE + z * BRICK_SIZE * BRICK_SIZE;
                            let generated = block_from_surface_height(
                                world_y,
                                surface_height,
                                dirt_depth,
                                surface_block,
                            );
                            let block =
                                overrides_or_generated(edits, world_x, world_y, world_z, generated);
                            voxels[idx] = block;
                            any_solid |= block.is_solid();
                        }
                    }
                }

                let brick_idx = bx
                    + by * PAGE_BRICKS_PER_AXIS
                    + bz * PAGE_BRICKS_PER_AXIS * PAGE_BRICKS_PER_AXIS;
                if any_solid {
                    occ |= 1u64 << brick_idx;
                }
                bricks.push(voxels);
            }
        }
    }

    BuiltPage {
        coord: page_coord,
        bricks,
        occ,
    }
}

fn sample_voxel_from_generator(
    generator: &TerrainGenerator,
    edits: &HashMap<WorldCoord, BlockId>,
    world_x: i64,
    world_y: i64,
    world_z: i64,
    voxel_size: i64,
) -> BlockId {
    if voxel_size <= 1 {
        return sample_base_voxel(generator, edits, world_x, world_y, world_z);
    }

    let child = voxel_size / 2;
    let mut children = [BlockId::AIR; 8];
    let mut idx = 0;
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let cx = world_x + dx * child;
                let cy = world_y + dy * child;
                let cz = world_z + dz * child;
                children[idx] = sample_base_voxel(generator, edits, cx, cy, cz);
                idx += 1;
            }
        }
    }

    downsample_voxel(&children)
}

fn sample_base_voxel(
    generator: &TerrainGenerator,
    edits: &HashMap<WorldCoord, BlockId>,
    world_x: i64,
    world_y: i64,
    world_z: i64,
) -> BlockId {
    edits
        .get(&WorldCoord {
            x: world_x,
            y: world_y,
            z: world_z,
        })
        .copied()
        .unwrap_or_else(|| generator.block_at_world(world_x, world_y, world_z))
}

fn overrides_or_generated(
    edits: &HashMap<WorldCoord, BlockId>,
    world_x: i64,
    world_y: i64,
    world_z: i64,
    generated: BlockId,
) -> BlockId {
    edits
        .get(&WorldCoord {
            x: world_x,
            y: world_y,
            z: world_z,
        })
        .copied()
        .unwrap_or(generated)
}

fn block_from_surface_height(
    world_y: i64,
    surface_height: i64,
    dirt_depth: i64,
    surface_block: BlockId,
) -> BlockId {
    if world_y > surface_height {
        BlockId::AIR
    } else if world_y == surface_height {
        surface_block
    } else if world_y > surface_height - dirt_depth {
        BlockId::DIRT
    } else {
        BlockId::STONE
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

fn mod_floor(value: i64, modulus: i64) -> i64 {
    let r = value % modulus;
    if r < 0 {
        r + modulus
    } else {
        r
    }
}

fn invalid_page_coord() -> [i32; 4] {
    [i32::MIN, i32::MIN, i32::MIN, 0]
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::generation::TerrainConfig;

    #[test]
    fn toroidal_index_wraps() {
        let gen = TerrainGenerator::new(TerrainConfig::default());
        let mut controller = ClipmapStreamingController::new(gen);

        controller.update(Vec3::new(0.0, 0.0, 0.0));
        let origin0 = controller.lod_origin(0);

        // Move by one page size along X.
        let page_size = PAGE_VOXELS_PER_AXIS as f32;
        controller.update(Vec3::new(page_size, 0.0, 0.0));
        let origin1 = controller.lod_origin(0);

        assert_ne!(origin0.x, origin1.x);
    }

    #[test]
    fn dirty_pages_with_small_shift() {
        let gen = TerrainGenerator::new(TerrainConfig::default());
        let mut controller = ClipmapStreamingController::new(gen);

        controller.update(Vec3::new(0.0, 0.0, 0.0));
        controller.take_dirty_state(); // clear initial dirty

        let page_size = PAGE_VOXELS_PER_AXIS as f32;
        controller.update(Vec3::new(page_size, 0.0, 0.0));
        let mut found = false;
        let mut dirty_count = 0;

        // Async builds may complete over multiple frames; pump updates until a slice lands.
        for _ in 0..64 {
            std::thread::sleep(Duration::from_millis(1));
            controller.update(Vec3::new(page_size, 0.0, 0.0));
            let dirty = controller.take_dirty_state();
            dirty_count = dirty.dirty_pages[0].len();
            if dirty_count > 0 {
                found = true;
                break;
            }
        }

        // Expect some dirty pages for LOD0, but not full rebuild (<= 16*16).
        assert!(
            found,
            "Expected at least one LOD0 page update after small shift"
        );
        assert!(dirty_count <= CLIPMAP_PAGE_GRID * CLIPMAP_PAGE_GRID);
    }

    #[test]
    fn lod0_renderable_before_full_ready() {
        let gen = TerrainGenerator::new(TerrainConfig::default());
        let mut controller = ClipmapStreamingController::new(gen);
        let camera = Vec3::new(0.0, 0.0, 0.0);

        controller.update(camera);

        let mut seen_renderable_before_ready = false;
        for _ in 0..256 {
            std::thread::sleep(Duration::from_millis(1));
            controller.update(camera);
            if controller.lod_renderable(0) && !controller.lod_ready(0) {
                seen_renderable_before_ready = true;
                break;
            }
        }

        assert!(
            seen_renderable_before_ready,
            "Expected LOD0 to become renderable before it is fully ready"
        );
    }

    #[test]
    fn pending_pages_prioritize_camera_proximity() {
        let gen = TerrainGenerator::new(TerrainConfig::default());
        let mut controller = ClipmapStreamingController::new(gen);
        let camera = WorldCoord {
            x: 10,
            y: 11,
            z: 12,
        };
        controller.camera_voxel = camera;

        let lod = 0;
        let voxel_size = controller.lod_voxel_size(lod);
        let page_size = PAGE_VOXELS_PER_AXIS as i64 * voxel_size;
        let coverage = controller.lod_coverage(lod);
        let origin = aligned_origin(camera, coverage, page_size);
        controller.enqueue_full_rebuild(lod, origin, voxel_size, page_size);

        let Some((_, first_coord, _, _)) = controller.pop_next_pending_page() else {
            panic!("Expected at least one pending page for LOD0");
        };
        assert_eq!(first_coord, (0, 0, 0));

        let mut previous_distance = page_distance_to_camera_sq(first_coord, camera, page_size);
        for _ in 0..32 {
            let Some((_, coord, _, _)) = controller.pop_next_pending_page() else {
                panic!("Expected enough pending pages to verify ordering");
            };
            let distance = page_distance_to_camera_sq(coord, camera, page_size);
            assert!(
                distance >= previous_distance,
                "Pending pages should be popped nearest-to-farthest ({} then {})",
                previous_distance,
                distance
            );
            previous_distance = distance;
        }
    }

    #[test]
    fn runtime_edit_overrides_generated_block() {
        let gen = TerrainGenerator::new(TerrainConfig::default());
        let mut controller = ClipmapStreamingController::new(gen);

        // Deep underground should be solid for generated terrain.
        let x = 0;
        let y = -128;
        let z = 0;
        assert!(controller.block_at_world(x, y, z).is_solid());

        assert!(controller.destroy_block_at_world(x, y, z));
        assert!(controller.block_at_world(x, y, z).is_air());

        assert!(controller.set_block_at_world(x, y, z, BlockId::STONE));
        assert_eq!(controller.block_at_world(x, y, z), BlockId::STONE);
    }
}
