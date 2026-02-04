//! Clipmap voxel storage with brick pools and page tables.
//!
//! Bricks are 8x8x8 voxels (512 total) encoded into palette16, palette32,
//! or raw16 entries. Headers are stored separately for fast traversal.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use voxelicous_core::types::BlockId;

/// Brick edge length in voxels.
pub const BRICK_SIZE: usize = 8;
/// Total voxels per brick.
pub const BRICK_VOXELS: usize = BRICK_SIZE * BRICK_SIZE * BRICK_SIZE;

/// Clipmap LOD count.
pub const CLIPMAP_LOD_COUNT: usize = 6;
/// Page grid size per axis.
pub const CLIPMAP_PAGE_GRID: usize = 16;
/// Bricks per page per axis (4x4x4 = 64 bricks).
pub const PAGE_BRICKS_PER_AXIS: usize = 4;
/// Total bricks per page.
pub const PAGE_BRICKS: usize = PAGE_BRICKS_PER_AXIS * PAGE_BRICKS_PER_AXIS * PAGE_BRICKS_PER_AXIS;
/// Voxels per page per axis (4 bricks * 8 voxels).
pub const PAGE_VOXELS_PER_AXIS: usize = PAGE_BRICKS_PER_AXIS * BRICK_SIZE;

/// Palette16 entry stride (bytes).
pub const PALETTE16_STRIDE: usize = 288;
/// Palette32 entry stride (bytes).
pub const PALETTE32_STRIDE: usize = 384;
/// Raw16 entry stride (bytes).
pub const RAW16_STRIDE: usize = 1024;

/// Newtype for brick identifiers (0 = empty).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct BrickId(pub u32);

/// Newtype for page identifiers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct PageId(pub u32);

/// Newtype for LOD level.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct LodLevel(pub u8);

/// Voxel-space coordinate (integer voxel units).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct VoxelCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// World-space voxel coordinate (integer voxel units in world space).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WorldCoord {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

/// Brick encoding type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrickEncoding {
    Palette16 = 0,
    Palette32 = 1,
    Raw16 = 2,
}

impl BrickEncoding {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Palette16),
            1 => Some(Self::Palette32),
            2 => Some(Self::Raw16),
            _ => None,
        }
    }
}

/// GPU brick header (32 bytes stride).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BrickHeader {
    pub palette_len: u8,
    pub encoding: u8,
    pub flags: u16,
    pub data_index: u32,
    pub occ_l0_lo: u32,
    pub occ_l0_hi: u32,
    pub occ_l1: u8,
    pub occ_l2: u8,
    pub _padding: u16,
    pub avg_color: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl Default for BrickHeader {
    fn default() -> Self {
        Self {
            palette_len: 0,
            encoding: BrickEncoding::Raw16 as u8,
            flags: 0,
            data_index: 0,
            occ_l0_lo: 0,
            occ_l0_hi: 0,
            occ_l1: 0,
            occ_l2: 0,
            _padding: 0,
            avg_color: 0,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// A single page containing 4x4x4 bricks.
#[derive(Clone, Debug)]
pub struct ClipmapPage {
    pub bricks: [BrickId; PAGE_BRICKS],
    pub occ: [u32; 2],
}

impl Default for ClipmapPage {
    fn default() -> Self {
        Self {
            bricks: [BrickId(0); PAGE_BRICKS],
            occ: [0, 0],
        }
    }
}

/// Encoded brick payload for CPU-side encoding/decoding.
#[derive(Clone, Debug)]
pub struct EncodedBrick {
    pub encoding: BrickEncoding,
    pub palette_len: u8,
    pub data: Vec<u8>,
}

/// Clipmap voxel store with brick headers and data pools.
#[derive(Debug, Default)]
pub struct ClipmapVoxelStore {
    headers: Vec<BrickHeader>,
    palette16_pool: Vec<u8>,
    palette32_pool: Vec<u8>,
    raw16_pool: Vec<u8>,
    free_headers: Vec<u32>,
    free_palette16: Vec<u32>,
    free_palette32: Vec<u32>,
    free_raw16: Vec<u32>,
}

impl ClipmapVoxelStore {
    /// Create a new clipmap voxel store.
    pub fn new() -> Self {
        let mut store = Self::default();
        // Reserve brick id 0 for empty bricks.
        store.headers.push(BrickHeader::default());
        store
    }

    /// Total number of brick headers (including empty).
    pub fn brick_count(&self) -> usize {
        self.headers.len()
    }

    /// Get a brick header by id.
    pub fn header(&self, id: BrickId) -> Option<&BrickHeader> {
        self.headers.get(id.0 as usize)
    }

    /// Encode and allocate a brick, returning its BrickId.
    pub fn allocate_brick(&mut self, voxels: &[BlockId; BRICK_VOXELS]) -> BrickId {
        if voxels.iter().all(|v| v.is_air()) {
            return BrickId(0);
        }

        let encoded = encode_brick(voxels);
        let (data_index, encoding) = match encoded.encoding {
            BrickEncoding::Palette16 => {
                let index = Self::allocate_pool_entry(
                    PALETTE16_STRIDE,
                    &mut self.palette16_pool,
                    &mut self.free_palette16,
                    &encoded.data,
                );
                (index, BrickEncoding::Palette16)
            }
            BrickEncoding::Palette32 => {
                let index = Self::allocate_pool_entry(
                    PALETTE32_STRIDE,
                    &mut self.palette32_pool,
                    &mut self.free_palette32,
                    &encoded.data,
                );
                (index, BrickEncoding::Palette32)
            }
            BrickEncoding::Raw16 => {
                let index = Self::allocate_pool_entry(
                    RAW16_STRIDE,
                    &mut self.raw16_pool,
                    &mut self.free_raw16,
                    &encoded.data,
                );
                (index, BrickEncoding::Raw16)
            }
        };

        let (occ_l0_lo, occ_l0_hi, occ_l1, occ_l2) = compute_occupancy(voxels);

        let header = BrickHeader {
            palette_len: encoded.palette_len,
            encoding: encoding as u8,
            flags: 0,
            data_index,
            occ_l0_lo,
            occ_l0_hi,
            occ_l1,
            occ_l2,
            _padding: 0,
            avg_color: 0,
            _pad0: 0,
            _pad1: 0,
        };

        if let Some(index) = self.free_headers.pop() {
            let idx = index as usize;
            if idx < self.headers.len() {
                self.headers[idx] = header;
                return BrickId(index);
            }
        }

        let id = BrickId(self.headers.len() as u32);
        self.headers.push(header);
        id
    }

    /// Free a brick, returning its pool entry to the free list.
    pub fn free_brick(&mut self, id: BrickId) {
        if id.0 == 0 {
            return;
        }
        let index = id.0 as usize;
        if index >= self.headers.len() {
            return;
        }
        let header = self.headers[index];
        if let Some(encoding) = BrickEncoding::from_u8(header.encoding) {
            match encoding {
                BrickEncoding::Palette16 => self.free_palette16.push(header.data_index),
                BrickEncoding::Palette32 => self.free_palette32.push(header.data_index),
                BrickEncoding::Raw16 => self.free_raw16.push(header.data_index),
            }
        }
        self.headers[index] = BrickHeader::default();
        self.free_headers.push(id.0);
    }

    /// Decode a brick into a dense voxel array.
    pub fn decode_brick(&self, id: BrickId) -> Option<[BlockId; BRICK_VOXELS]> {
        let header = *self.header(id)?;
        let encoding = BrickEncoding::from_u8(header.encoding)?;
        let data = match encoding {
            BrickEncoding::Palette16 => {
                self.pool_entry(&self.palette16_pool, PALETTE16_STRIDE, header.data_index)?
            }
            BrickEncoding::Palette32 => {
                self.pool_entry(&self.palette32_pool, PALETTE32_STRIDE, header.data_index)?
            }
            BrickEncoding::Raw16 => {
                self.pool_entry(&self.raw16_pool, RAW16_STRIDE, header.data_index)?
            }
        };

        Some(decode_brick(encoding, header.palette_len, data))
    }

    /// Get the raw header buffer for GPU upload.
    pub fn headers(&self) -> &[BrickHeader] {
        &self.headers
    }

    /// Get the palette16 pool as raw bytes.
    pub fn palette16_pool(&self) -> &[u8] {
        &self.palette16_pool
    }

    /// Get the palette32 pool as raw bytes.
    pub fn palette32_pool(&self) -> &[u8] {
        &self.palette32_pool
    }

    /// Get the raw16 pool as raw bytes.
    pub fn raw16_pool(&self) -> &[u8] {
        &self.raw16_pool
    }

    fn allocate_pool_entry(
        stride: usize,
        pool: &mut Vec<u8>,
        free_list: &mut Vec<u32>,
        data: &[u8],
    ) -> u32 {
        debug_assert_eq!(data.len(), stride);
        if let Some(index) = free_list.pop() {
            let offset = index as usize * stride;
            pool[offset..offset + stride].copy_from_slice(data);
            index
        } else {
            let index = (pool.len() / stride) as u32;
            pool.extend_from_slice(data);
            index
        }
    }

    fn pool_entry<'a>(&self, pool: &'a [u8], stride: usize, index: u32) -> Option<&'a [u8]> {
        let offset = index as usize * stride;
        if offset + stride > pool.len() {
            return None;
        }
        Some(&pool[offset..offset + stride])
    }
}

/// Compute occupancy masks for a brick.
pub fn compute_occupancy(voxels: &[BlockId; BRICK_VOXELS]) -> (u32, u32, u8, u8) {
    let mut occ_l0: u64 = 0;
    let mut occ_l1: u8 = 0;
    let mut any = false;

    // 4x4x4 occupancy from 2x2x2 voxel groups.
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                let mut solid = false;
                for dz in 0..2 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let vx = x * 2 + dx;
                            let vy = y * 2 + dy;
                            let vz = z * 2 + dz;
                            let idx = vx + vy * BRICK_SIZE + vz * BRICK_SIZE * BRICK_SIZE;
                            if voxels[idx].is_solid() {
                                solid = true;
                                any = true;
                                break;
                            }
                        }
                        if solid {
                            break;
                        }
                    }
                    if solid {
                        break;
                    }
                }

                if solid {
                    let bit = (x + y * 4 + z * 16) as u64;
                    occ_l0 |= 1u64 << bit;
                }
            }
        }
    }

    // 2x2x2 occupancy from 4x4x4 groups.
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let mut solid = false;
                for dz in 0..2 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let sx = x * 2 + dx;
                            let sy = y * 2 + dy;
                            let sz = z * 2 + dz;
                            let bit = (sx + sy * 4 + sz * 16) as u64;
                            if (occ_l0 >> bit) & 1 == 1 {
                                solid = true;
                                break;
                            }
                        }
                        if solid {
                            break;
                        }
                    }
                    if solid {
                        break;
                    }
                }

                if solid {
                    let bit = (x + y * 2 + z * 4) as u8;
                    occ_l1 |= 1u8 << bit;
                }
            }
        }
    }

    let occ_l2 = if any { 1 } else { 0 };
    let occ_l0_lo = (occ_l0 & 0xFFFF_FFFF) as u32;
    let occ_l0_hi = (occ_l0 >> 32) as u32;

    (occ_l0_lo, occ_l0_hi, occ_l1, occ_l2)
}

/// Encode a brick into palette/raw data.
pub fn encode_brick(voxels: &[BlockId; BRICK_VOXELS]) -> EncodedBrick {
    let mut counts: HashMap<BlockId, usize> = HashMap::new();
    for v in voxels {
        *counts.entry(*v).or_insert(0) += 1;
    }

    let mut palette: Vec<(BlockId, usize)> = counts.into_iter().collect();
    palette.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0 .0.cmp(&b.0 .0)));

    let palette_len = palette.len();

    if palette_len <= 16 {
        let mut palette_ids = [BlockId::AIR; 16];
        for (i, (id, _)) in palette.iter().enumerate() {
            palette_ids[i] = *id;
        }
        let data = encode_palette16(voxels, &palette_ids);
        EncodedBrick {
            encoding: BrickEncoding::Palette16,
            palette_len: palette_len as u8,
            data,
        }
    } else if palette_len <= 32 {
        let mut palette_ids = [BlockId::AIR; 32];
        for (i, (id, _)) in palette.iter().enumerate() {
            palette_ids[i] = *id;
        }
        let data = encode_palette32(voxels, &palette_ids);
        EncodedBrick {
            encoding: BrickEncoding::Palette32,
            palette_len: palette_len as u8,
            data,
        }
    } else {
        let data = encode_raw16(voxels);
        EncodedBrick {
            encoding: BrickEncoding::Raw16,
            palette_len: 0,
            data,
        }
    }
}

/// Decode a brick from palette/raw data.
pub fn decode_brick(
    encoding: BrickEncoding,
    palette_len: u8,
    data: &[u8],
) -> [BlockId; BRICK_VOXELS] {
    match encoding {
        BrickEncoding::Palette16 => decode_palette16(data),
        BrickEncoding::Palette32 => decode_palette32(data, palette_len),
        BrickEncoding::Raw16 => decode_raw16(data),
    }
}

fn encode_palette16(voxels: &[BlockId; BRICK_VOXELS], palette: &[BlockId; 16]) -> Vec<u8> {
    let mut data = vec![0u8; PALETTE16_STRIDE];
    // Palette first (16 * u16)
    for (i, id) in palette.iter().enumerate() {
        let bytes = id.0.to_le_bytes();
        let offset = i * 2;
        data[offset] = bytes[0];
        data[offset + 1] = bytes[1];
    }

    // Index mapping for quick lookup
    let mut map: HashMap<BlockId, u8> = HashMap::new();
    for (i, id) in palette.iter().enumerate() {
        map.insert(*id, i as u8);
    }

    // Indices (4 bits each)
    let mut out_idx = 32;
    for chunk in voxels.chunks(2) {
        let idx0 = *map.get(&chunk[0]).unwrap_or(&0);
        let idx1 = *map.get(&chunk[1]).unwrap_or(&0);
        data[out_idx] = (idx0 & 0x0F) | ((idx1 & 0x0F) << 4);
        out_idx += 1;
    }

    data
}

fn encode_palette32(voxels: &[BlockId; BRICK_VOXELS], palette: &[BlockId; 32]) -> Vec<u8> {
    let mut data = vec![0u8; PALETTE32_STRIDE];
    // Palette first (32 * u16)
    for (i, id) in palette.iter().enumerate() {
        let bytes = id.0.to_le_bytes();
        let offset = i * 2;
        data[offset] = bytes[0];
        data[offset + 1] = bytes[1];
    }

    // Index mapping
    let mut map: HashMap<BlockId, u8> = HashMap::new();
    for (i, id) in palette.iter().enumerate() {
        map.insert(*id, i as u8);
    }

    let mut bit_cursor = 0usize;
    let base = 64;
    for v in voxels {
        let idx = *map.get(v).unwrap_or(&0) & 0x1F;
        let byte_idx = base + (bit_cursor >> 3);
        let bit_off = (bit_cursor & 7) as u8;

        let low = idx << bit_off;
        data[byte_idx] |= low;
        if bit_off > 3 {
            if byte_idx + 1 < data.len() {
                data[byte_idx + 1] |= idx >> (8 - bit_off);
            }
        }

        bit_cursor += 5;
    }

    data
}

fn encode_raw16(voxels: &[BlockId; BRICK_VOXELS]) -> Vec<u8> {
    let mut data = vec![0u8; RAW16_STRIDE];
    let mut offset = 0;
    for v in voxels {
        let bytes = v.0.to_le_bytes();
        data[offset] = bytes[0];
        data[offset + 1] = bytes[1];
        offset += 2;
    }
    data
}

fn decode_palette16(data: &[u8]) -> [BlockId; BRICK_VOXELS] {
    let mut palette = [BlockId::AIR; 16];
    for i in 0..16 {
        let offset = i * 2;
        palette[i] = BlockId(u16::from_le_bytes([data[offset], data[offset + 1]]));
    }

    let mut out = [BlockId::AIR; BRICK_VOXELS];
    let mut out_idx = 0;
    for &byte in &data[32..] {
        let idx0 = byte & 0x0F;
        let idx1 = (byte >> 4) & 0x0F;
        out[out_idx] = palette[idx0 as usize];
        out[out_idx + 1] = palette[idx1 as usize];
        out_idx += 2;
    }
    out
}

fn decode_palette32(data: &[u8], palette_len: u8) -> [BlockId; BRICK_VOXELS] {
    let mut palette = [BlockId::AIR; 32];
    for i in 0..32 {
        let offset = i * 2;
        palette[i] = BlockId(u16::from_le_bytes([data[offset], data[offset + 1]]));
    }
    let max_index = palette_len.min(32) as u8;

    let mut out = [BlockId::AIR; BRICK_VOXELS];
    let base = 64;
    let mut bit_cursor = 0usize;
    for i in 0..BRICK_VOXELS {
        let byte_idx = base + (bit_cursor >> 3);
        let bit_off = (bit_cursor & 7) as u8;
        let low = data[byte_idx] as u16;
        let high = if byte_idx + 1 < data.len() {
            (data[byte_idx + 1] as u16) << 8
        } else {
            0
        };
        let raw = low | high;
        let mut idx = ((raw >> bit_off) & 0x1F) as u8;
        if idx >= max_index {
            idx = 0;
        }
        out[i] = palette[idx as usize];
        bit_cursor += 5;
    }
    out
}

fn decode_raw16(data: &[u8]) -> [BlockId; BRICK_VOXELS] {
    let mut out = [BlockId::AIR; BRICK_VOXELS];
    for i in 0..BRICK_VOXELS {
        let offset = i * 2;
        out[i] = BlockId(u16::from_le_bytes([data[offset], data[offset + 1]]));
    }
    out
}

/// Downsample a 2x2x2 voxel block into one voxel.
pub fn downsample_voxel(children: &[BlockId; 8]) -> BlockId {
    let mut counts: HashMap<BlockId, usize> = HashMap::new();
    let mut solid_count = 0;
    let mut has_air = false;
    let mut has_grass = false;
    for v in children {
        if v.is_solid() {
            solid_count += 1;
            *counts.entry(*v).or_insert(0) += 1;
            has_grass |= *v == BlockId::GRASS;
        } else {
            has_air = true;
        }
    }

    if solid_count < 2 {
        return BlockId::AIR;
    }

    // Preserve thin grass shells on coarse LODs where air+surface blocks mix.
    if has_air && has_grass {
        return BlockId::GRASS;
    }

    let mut best = BlockId::AIR;
    let mut best_count = 0;
    for (id, count) in counts {
        if count > best_count || (count == best_count && id.0 < best.0) {
            best = id;
            best_count = count;
        }
    }

    best
}

/// Downsample a dense volume by 2x along each axis.
pub fn downsample_volume_2x(input: &[BlockId], size: usize) -> Vec<BlockId> {
    assert!(size % 2 == 0, "Input size must be even");
    let out_size = size / 2;
    let mut output = vec![BlockId::AIR; out_size * out_size * out_size];

    for z in 0..out_size {
        for y in 0..out_size {
            for x in 0..out_size {
                let mut children = [BlockId::AIR; 8];
                let mut idx = 0;
                for dz in 0..2 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let sx = x * 2 + dx;
                            let sy = y * 2 + dy;
                            let sz = z * 2 + dz;
                            let in_idx = sx + sy * size + sz * size * size;
                            children[idx] = input[in_idx];
                            idx += 1;
                        }
                    }
                }
                let out_idx = x + y * out_size + z * out_size * out_size;
                output[out_idx] = downsample_voxel(&children);
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brick_header_size() {
        assert_eq!(std::mem::size_of::<BrickHeader>(), 32);
    }

    #[test]
    fn palette16_roundtrip() {
        let mut voxels = [BlockId::AIR; BRICK_VOXELS];
        for i in 0..BRICK_VOXELS {
            voxels[i] = if i % 7 == 0 {
                BlockId::STONE
            } else {
                BlockId::AIR
            };
        }
        let encoded = encode_brick(&voxels);
        assert_eq!(encoded.encoding, BrickEncoding::Palette16);
        let decoded = decode_brick(encoded.encoding, encoded.palette_len, &encoded.data);
        assert_eq!(decoded[..], voxels[..]);
    }

    #[test]
    fn palette32_roundtrip() {
        let mut voxels = [BlockId::AIR; BRICK_VOXELS];
        for i in 0..BRICK_VOXELS {
            voxels[i] = BlockId((i % 31 + 1) as u16);
        }
        let encoded = encode_brick(&voxels);
        assert_eq!(encoded.encoding, BrickEncoding::Palette32);
        let decoded = decode_brick(encoded.encoding, encoded.palette_len, &encoded.data);
        assert_eq!(decoded[..], voxels[..]);
    }

    #[test]
    fn raw16_roundtrip() {
        let mut voxels = [BlockId::AIR; BRICK_VOXELS];
        for i in 0..BRICK_VOXELS {
            voxels[i] = BlockId((i as u16) + 1);
        }
        let encoded = encode_brick(&voxels);
        assert_eq!(encoded.encoding, BrickEncoding::Raw16);
        let decoded = decode_brick(encoded.encoding, encoded.palette_len, &encoded.data);
        assert_eq!(decoded[..], voxels[..]);
    }

    #[test]
    fn occupancy_masks() {
        let mut voxels = [BlockId::AIR; BRICK_VOXELS];
        voxels[0] = BlockId::STONE;
        voxels[BRICK_VOXELS - 1] = BlockId::STONE;

        let (lo, hi, l1, l2) = compute_occupancy(&voxels);
        assert!(lo != 0 || hi != 0);
        assert!(l1 != 0);
        assert_eq!(l2, 1);
    }

    #[test]
    fn downsample_rule() {
        let children = [
            BlockId::AIR,
            BlockId::STONE,
            BlockId::STONE,
            BlockId::AIR,
            BlockId::AIR,
            BlockId::AIR,
            BlockId::AIR,
            BlockId::AIR,
        ];
        let out = downsample_voxel(&children);
        assert_eq!(out, BlockId::STONE);
    }

    #[test]
    fn downsample_preserves_surface_grass() {
        let children = [
            BlockId::AIR,
            BlockId::AIR,
            BlockId::GRASS,
            BlockId::DIRT,
            BlockId::STONE,
            BlockId::STONE,
            BlockId::AIR,
            BlockId::AIR,
        ];
        let out = downsample_voxel(&children);
        assert_eq!(out, BlockId::GRASS);
    }
}
