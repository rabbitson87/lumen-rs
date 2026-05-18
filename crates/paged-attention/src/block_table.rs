//! Page Table — Per-sequence logical-to-physical block mapping.
//!
//! Each sequence (request) has its own PageTable that maps logical block
//! indices (0, 1, 2, ...) to physical block IDs in the BlockAllocator's pool.
//!
//! With per-layer Metal buffers, a single physical block ID represents a slot
//! that exists in ALL layer buffers simultaneously. So the page table is a
//! simple `Vec<u32>` — shared across all layers.

use anyhow::{Result, bail};

use crate::block_allocator::BlockAllocator;

/// Per-sequence page table mapping logical blocks to physical block IDs.
///
/// One block ID indexes into every layer's buffer at the same position.
/// `table[logical_block_idx]` = physical block ID (same for all layers).
pub struct PageTable {
    /// `table[logical_block_idx]` → physical block ID.
    table: Vec<u32>,
}

impl PageTable {
    /// Create an empty page table.
    pub fn new() -> Self {
        Self { table: Vec::new() }
    }

    /// Number of logical blocks currently mapped.
    pub fn num_blocks(&self) -> usize {
        self.table.len()
    }

    /// Append one new block.
    ///
    /// Allocates ONE physical block ID from the allocator.
    /// This ID is valid across all per-layer buffers.
    pub fn append_block(&mut self, allocator: &mut BlockAllocator) -> Result<()> {
        match allocator.allocate() {
            Some(block_id) => {
                self.table.push(block_id);
                Ok(())
            }
            None => bail!("OOM: no free blocks"),
        }
    }

    /// Free all physical blocks back to the allocator.
    pub fn free_all(&mut self, allocator: &mut BlockAllocator) {
        for &block_id in &self.table {
            allocator.free(block_id);
        }
        self.table.clear();
    }

    /// Get the physical block ID for a logical index.
    pub fn get(&self, logical_idx: usize) -> u32 {
        self.table[logical_idx]
    }

    /// Get the full block table (for Metal kernel input).
    pub fn as_slice(&self) -> &[u32] {
        &self.table
    }

    /// Block table padded to `max_blocks` length.
    ///
    /// Used when building batched block_tables tensor:
    /// `block_tables[batch_idx][0..max_blocks]`
    pub fn padded(&self, max_blocks: usize) -> Vec<u32> {
        let mut padded = vec![0u32; max_blocks];
        let copy_len = self.table.len().min(max_blocks);
        padded[..copy_len].copy_from_slice(&self.table[..copy_len]);
        padded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_allocator::BlockConfig;
    use candle_metal_kernels::metal::Device;

    #[test]
    fn test_page_table_lifecycle() {
        let device = match Device::system_default() {
            Some(d) => d,
            None => return,
        };
        let config = BlockConfig::uniform(16, 3, 2, 64);
        let budget = config.block_bytes_total() * 20;
        let mut alloc = BlockAllocator::new(&device, budget, config).unwrap();
        let initial_free = alloc.num_free_blocks();

        let mut pt = PageTable::new();
        assert_eq!(pt.num_blocks(), 0);

        // Append 2 logical blocks → 2 physical blocks (shared across layers)
        pt.append_block(&mut alloc).unwrap();
        pt.append_block(&mut alloc).unwrap();
        assert_eq!(pt.num_blocks(), 2);
        assert_eq!(alloc.num_free_blocks(), initial_free - 2);

        // Block IDs
        let b0 = pt.get(0);
        let b1 = pt.get(1);
        assert_ne!(b0, b1);

        // Slice
        assert_eq!(pt.as_slice().len(), 2);

        // Padded
        let padded = pt.padded(5);
        assert_eq!(padded.len(), 5);
        assert_eq!(padded[0], b0);
        assert_eq!(padded[1], b1);
        assert_eq!(padded[2], 0); // padding

        // Free all
        pt.free_all(&mut alloc);
        assert_eq!(pt.num_blocks(), 0);
        assert_eq!(alloc.num_free_blocks(), initial_free);
    }
}
