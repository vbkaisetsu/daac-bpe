//! # daac-bpe
//!
//! A BPE tokenizer that operates in linear time relative to the input length.
//!
//! While this implementation is based on the following paper, various optimizations have been
//! applied to improve performance.
//!
//! This implementation is based on the following paper:
//!
//! > Shenghu Jiang and Ruihao Gong. *Incremental BPE Tokenization.*
//! > ICML 2026. <https://arxiv.org/abs/2605.30813>
//!
//! # Examples
//!
//! A vocabulary is an ordered list of token byte strings, and a token's rank is its position in
//! the list. All 256 single-byte tokens must be present; merge rules follow in priority order.
//!
//! ```
//! use daac_bpe::IncrementalBpe;
//!
//! let mut vocab: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
//! vocab.push(b"ab".to_vec()); // rank 256
//! vocab.push(b"abc".to_vec()); // rank 257
//!
//! let bpe = IncrementalBpe::new(&vocab)?;
//!
//! // Tokens are appended, so one buffer can be reused across chunks.
//! let mut tokens = vec![];
//! bpe.encode(b"abcab", &mut tokens);
//! assert_eq!(tokens, [257, 256]);
//! # Ok::<(), daac_bpe::Error>(())
//! ```
//!
//! To optimize the memory layout for a sample of the text to be tokenized, build the tokenizer
//! with [`IncrementalBpeBuilder::corpus()`].
#![no_std]

#[cfg(not(any(target_pointer_width = "32", target_pointer_width = "64")))]
compile_error!("`target_pointer_width` must be 32 or 64");

extern crate alloc;

mod builder;
mod error;
mod sufsuc;
mod utils;
mod vocab;

use alloc::vec;
use alloc::vec::Vec;

use daachorse::DoubleArrayAhoCorasick;

pub use crate::builder::IncrementalBpeBuilder;
pub use crate::error::{Error, Result};
use crate::sufsuc::CentroidSearchTree;
use crate::utils::FromU32;
use crate::vocab::{INVALID_ID, StateEntry, TokenIndex, TokenTable};

/// An incremental BPE tokenizer.
pub struct IncrementalBpe {
    table: TokenTable,
    pma: DoubleArrayAhoCorasick<StateEntry>,
    centroid_trees: CentroidSearchTree,
    index: TokenIndex,
}

impl IncrementalBpe {
    /// Creates a new BPE tokenizer.
    ///
    /// The vocabulary must be *proper*: for every canonical rule, the ranks of both components must
    /// be strictly smaller than that of the merged token. Non-canonical entries (`T_D(t) != [t]`)
    /// are dropped silently.
    ///
    /// # Arguments
    ///
    /// * `tokens` - List of token byte strings in priority order. A token's position is its rank.
    ///
    /// # Examples
    ///
    /// ```
    /// use daac_bpe::IncrementalBpe;
    ///
    /// let bpe = IncrementalBpe::new((0u8..=255).map(|b| [b]))?;
    /// let mut tokens = vec![];
    /// bpe.encode(b"hi", &mut tokens);
    /// assert_eq!(tokens, [u32::from(b'h'), u32::from(b'i')]);
    /// # Ok::<(), daac_bpe::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// * [`Error::EmptyToken`] - A token of zero-length is present.
    /// * [`Error::DuplicateToken`] - The same byte string appears twice.
    /// * [`Error::VocabTooLarge`] - The token count or the total byte length does not fit in a `u32`.
    /// * [`Error::MissingByteToken`] - A single-byte token is missing.
    /// * [`Error::NotProper`] - The dictionary is not proper.
    /// * [`Error::TokenTooLong`] - A token longer than `u16::MAX` is present.
    /// * [`Error::Automaton`] - Building the automaton failed.
    pub fn new<I, P>(tokens: I) -> Result<Self>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        IncrementalBpeBuilder::new().build(tokens)
    }

    /// Counts how many times each Centroid Search Tree node is visited while encoding `corpus`.
    ///
    /// Used to learn the layout for [`IncrementalBpeBuilder::corpus`]. One element is one chunk.
    ///
    /// # Arguments
    ///
    /// * `corpus` - List of sample chunks.
    fn count_cst_visits(&self, corpus: &[Vec<u8>]) -> Vec<u64> {
        let mut counts = vec![0; self.centroid_trees.nodes.len()];
        let mut scratch = vec![];
        for piece in corpus {
            scratch.clear();
            self.encode_visit(piece, &mut scratch, &mut |idx| {
                counts[usize::from_u32(idx)] += 1;
            });
        }
        counts
    }

    /// Tokenizes one chunk and appends the token sequence to `out`.
    ///
    /// Equivalent to `T_D(piece)`. The per-byte cost is O(log² t) regardless of the chunk length,
    /// so the whole input may be passed as a single chunk. The history borrows the tail of `out`
    /// while a chunk is consumed, so reusing one `Vec` across chunks removes the per-chunk
    /// allocation. Tokens are written only once the chunk is complete.
    ///
    /// # Arguments
    ///
    /// * `piece` - Chunk to tokenize, the unit a pre-tokenizer delivers to the BPE stage.
    /// * `out` - Destination the tokens are appended to.
    ///
    /// # Examples
    ///
    /// ```
    /// use daac_bpe::IncrementalBpe;
    ///
    /// let bpe = IncrementalBpe::new((0u8..=255).map(|b| [b]))?;
    /// let mut tokens = vec![];
    /// bpe.encode(b"hi", &mut tokens);
    /// bpe.encode(b"!", &mut tokens);
    /// assert_eq!(tokens, [u32::from(b'h'), u32::from(b'i'), u32::from(b'!')]);
    /// # Ok::<(), daac_bpe::Error>(())
    /// ```
    pub fn encode(&self, piece: &[u8], out: &mut Vec<u32>) {
        self.encode_visit(piece, out, &mut |_| {});
    }

    /// [`encode`](Self::encode) with the CST profiling hook exposed.
    ///
    /// Reserves the history at the tail of `out` and passes it on as a `(buf, base)` pair, where
    /// `buf[base]` is the ε sentinel `INVALID_ID` and `buf[base + j]` is `dfs_in(θ(s[1..=j]))`, so
    /// `H(ℓ)` is `buf[buf.len() - ℓ]`. `INVALID_ID` exceeds every timestamp, so ε fails every
    /// interval test without a special case. Timestamps are stored rather than token ids;
    /// `emit_tokens` maps them back.
    ///
    /// # Arguments
    ///
    /// * `piece` - Chunk to tokenize.
    /// * `out` - Destination for the tokens, and the history while the chunk is consumed.
    /// * `visit` - Hook receiving every CST node touched.
    #[inline]
    fn encode_visit(&self, piece: &[u8], out: &mut Vec<u32>, visit: &mut impl FnMut(u32)) {
        if piece.is_empty() {
            return;
        }

        // If the chunk is itself a canonical token then T_D(t) == [t], so that single token is
        // the answer.
        if let Some(rank) = self.index.get(piece) {
            out.push(rank);
            return;
        }

        // One slot per byte plus the ε sentinel, so the per-byte loop never reallocates.
        out.reserve(piece.len() + 1);
        let base = out.len();
        out.push(INVALID_ID);
        self.consume(piece, out, base, visit);
        self.emit_tokens(out, base);
    }

    /// Consumes a whole chunk and fills the history.
    ///
    /// `find_overlapping_no_suffix_iter` yields the single longest pattern ending at each position,
    /// which is exactly τ(sc). Every single byte is in V̄, so it yields one match per input byte.
    ///
    /// # Arguments
    ///
    /// * `piece` - Chunk to consume, which is the entire input as far as the automaton is concerned.
    /// * `buf`, `base` - Where the history lives. See [`encode_visit`](Self::encode_visit).
    /// * `visit` - CST profiling hook.
    fn consume(&self, piece: &[u8], buf: &mut Vec<u32>, base: usize, visit: &mut impl FnMut(u32)) {
        for m in self.pma.find_overlapping_no_suffix_iter(piece) {
            self.advance(m.value(), buf, base, visit);
        }
    }

    /// Computes θ for one byte from the payload of τ and appends it to the history.
    ///
    /// Tests the Prefix Last-Token Condition on τ itself first. If τ satisfies it then τ is the
    /// deepest node of SufSucTree(τ) and is the answer. Otherwise it descends the Centroid Search
    /// Tree.
    ///
    /// # Arguments
    ///
    /// * `entry` - Payload of τ, as yielded by the iterator in [`consume`](Self::consume).
    /// * `buf`, `base` - Where the history lives. See [`encode_visit`](Self::encode_visit).
    /// * `visit` - CST profiling hook.
    #[inline]
    fn advance(
        &self,
        entry: StateEntry,
        buf: &mut Vec<u32>,
        base: usize,
        visit: &mut impl FnMut(u32),
    ) {
        // n = |sc|. The history holds one entry per consumed byte plus the ε sentinel.
        let n = buf.len() - base;
        let tau_ok = entry.suc_len == 0
            || entry
                .valid
                .contains(&buf[base + n - usize::from(entry.suc_len)]);
        let dfs_in = if tau_ok {
            entry.dfs_in_tau
        } else {
            // `descend` indexes the history from its own end, so it does not need `base`.
            let last = self.descend(entry.dfs_in_tau, &buf[base..], visit);
            self.table.dfs_in[usize::from_u32(last)]
        };
        buf.push(dfs_in);
    }

    /// Descends the Centroid Search Tree of SufSucTree(τ) and returns the token id of the new last
    /// token θ(sc).
    ///
    /// The nodes satisfying the Prefix Last-Token Condition form a single path from the root, so
    /// each centroid needs only an O(1) interval test. On failure the search moves to the component
    /// on the parent side, and on success it binary-searches the children and descends. That gives
    /// O(log² |τ|) per byte.
    ///
    /// # Arguments
    ///
    /// * `dfs_in_tau` - DFS timestamp of τ, the value carried by the payload.
    /// * `dfs` - The chunk's part of the history, so `H(ℓ)` is `dfs[dfs.len() - ℓ]`.
    /// * `visit` - CST profiling hook.
    fn descend(&self, dfs_in_tau: u32, dfs: &[u32], visit: &mut impl FnMut(u32)) -> u32 {
        let mut idx = self.centroid_trees.roots[usize::from_u32(dfs_in_tau)];
        loop {
            visit(idx);
            let node = self.centroid_trees.nodes[usize::from_u32(idx)];
            // The interval is inlined into the node, so the test touches no other array.
            let search = node.search;
            // The Prefix Last-Token Condition.
            let suc_len = usize::from(search.suc_len);
            let valid = search.suc_len == 0 || search.valid.contains(&dfs[dfs.len() - suc_len]);

            if !valid {
                // The root is an atomic token and always valid, so an invalid node always has an
                // upward component to fall back to.
                idx = node.up_component;
            } else {
                // A child v of u satisfies suc(v) = u, so its test queries H(|u|).
                let len = usize::from(search.len);
                let h_dfs_in = dfs[dfs.len() - len];
                let start = usize::from_u32(node.child_start);
                let children =
                    &self.centroid_trees.children[start..start + usize::from_u32(node.child_len)];
                // The intervals are disjoint and sorted, so the only candidate is the last child
                // whose start is at most H(|u|).
                let pos = children.partition_point(|child| child.valid.start <= h_dfs_in);
                if pos > 0 && children[pos - 1].valid.contains(&h_dfs_in) {
                    // A removed child is an ancestor in the CST, which the path never revisits.
                    idx = children[pos - 1].node;
                } else {
                    return node.token;
                }
            }
        }
    }

    /// Replaces the history in place with the tokens of the chunk.
    ///
    /// Unrolls the recursion `T(s) = T(s_pre) ⊕ [θ(s)]`, emitting θ(s) and stepping back |θ(s)|
    /// bytes until the virtual root.
    ///
    /// Both indices move right to left, and the read index advances by |θ| >= 1 per token while the
    /// write index advances by one, so the write index never passes the read index and overwriting
    /// what the traversal has already read is safe. The result ends up packed at the right end,
    /// which one `copy_within` shifts down to `base`.
    ///
    /// # Arguments
    ///
    /// * `buf`, `base` - Where the history lives. See [`encode_visit`](Self::encode_visit).
    fn emit_tokens(&self, buf: &mut Vec<u32>, base: usize) {
        let table = &self.table;
        // The last entry is θ of the whole chunk; `base` holds the sentinel where it stops.
        let end = buf.len() - 1;
        let mut read = end;
        let mut write = end;
        while read > base {
            let entry = table.dfs_in_inv[usize::from_u32(buf[read])];
            buf[write] = entry.rank;
            write -= 1;
            read -= usize::from(entry.len);
        }
        let tokens = end - write;
        buf.copy_within(write + 1..end + 1, base);
        buf.truncate(base + tokens);
    }
}
