//! The normalized vocabulary V̄, the linearization of the Successor Forest, and the tables
//! supporting them.
//!
//! [`TokenTableBuilder::new`] normalizes the vocabulary, keeping only the canonical tokens and
//! recovering the canonical rule `(pre(t), suc(t)) -> t`, then linearizes the forest with a DFS.
//! [`TokenTableBuilder::finish`] leaves only what is read at run time in `TokenTable`.

use alloc::vec;
use alloc::vec::Vec;
use core::hash::Hasher;
use core::range::Range;

use hashbrown::HashMap;
use rustc_hash::{FxBuildHasher, FxHasher};

use crate::utils::FromU32;
use crate::{Error, Result};

/// The value used to indicate an invalid node ID.
pub(crate) const INVALID_ID: u32 = u32::MAX;

/// Hash map used only during construction.
pub type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

/// The search entry required to test the Prefix Last-Token Condition for a token `t` with a single
/// read.
#[derive(Clone, Copy, Default)]
pub(crate) struct SearchEntry {
    pub valid: Range<u32>,
    pub len: u16,
    pub suc_len: u16,
}

/// The payload carried by the Aho-Corasick automaton, i.e. what the automaton returns about the
/// longest suffix token τ(sc).
///
/// This struct does not carry τ's own token id. That id is cold, and the CST root can be looked up
/// from `dfs_in_tau`, so leaving it out is what keeps the entry at 16 bytes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct StateEntry {
    pub valid: Range<u32>,
    pub dfs_in_tau: u32,
    pub len: u16,
    pub suc_len: u16,
}
const _: () = assert!(size_of::<StateEntry>() == 16);

/// The inverse of the DFS linearization. It looks up the rank and byte length of `t` from
/// `dfs_in(t)`.
#[derive(Clone, Copy, Default)]
pub(crate) struct DfsEntry {
    pub rank: u32,
    pub len: u16,
}
const _: () = assert!(size_of::<DfsEntry>() == 8);

/// The token table used in the tokenization. This struct is a part of [`TokenTableBuilder`].
pub(crate) struct TokenTable {
    pub dfs_in: Vec<u32>,
    pub dfs_in_inv: Vec<DfsEntry>,
}

/// The normalized vocabulary under construction, as a struct of arrays indexed by a dense internal
/// token id.
pub(crate) struct TokenTableBuilder {
    /// The byte strings of all tokens, concatenated. Sliced with `offsets`.
    pub bytes: Vec<u8>,
    /// `offsets[i]..offsets[i + 1]` is the range of token `i` within `bytes`.
    pub offsets: Vec<u32>,
    /// The lengths of tokens.
    pub token_len: Vec<u16>,
    /// The externally visible rank, i.e. the merge priority.
    pub rank: Vec<u32>,
    /// pre(t) from t's canonical rule. `INVALID_ID` for atomic tokens.
    pub pre: Vec<u32>,
    /// suc(t) from t's canonical rule, i.e. t's parent in the Successor Forest. `INVALID_ID` for
    /// atomic tokens.
    pub suc: Vec<u32>,
    /// Entry timestamps of the DFS over the Successor Forest.
    pub dfs_in: Vec<u32>,
    /// Exit timestamps. The subtree of `t` is exactly `[dfs_in(t), dfs_out(t))`.
    pub dfs_out: Vec<u32>,
    /// The valid interval and lengths of each token.
    pub search: Vec<SearchEntry>,
    /// The inverse of `dfs_in`.
    pub dfs_in_inv: Vec<DfsEntry>,
}

/// A hash table mapping a token's byte string to its rank, for the whole-chunk
/// fast path.
///
/// This is not discussed in the original paper. When a pre-tokenized chunk happens to be a single
/// token, one lookup gives the token ID.
///
/// Keys of up to `INLINE_MAX` bytes are packed into a single `u64` and held in
/// `short`; longer keys are held by value in `long`.
pub(crate) struct TokenIndex {
    /// Keys of at most `INLINE_MAX` bytes, packed by [`Self::padded_word`].
    short: FxHashMap<u64, u32>,
    /// Longer keys, owned and compared as byte strings. Cold path.
    long: FxHashMap<Vec<u8>, u32>,
    /// max |t| over all of V̄, so that [`Self::get`] can reject chunks that are
    /// too long before hashing.
    max_len: u16,
}

const INLINE_MAX: usize = 8;

/// The padding byte used to erase a short key's length. 0xF5 never occurs in
/// valid UTF-8.
const PAD: u8 = 0xF5;
const PAD_WORD: u64 = u64::from_ne_bytes([PAD; INLINE_MAX]);
const SWAR_LO: u64 = 0x0101_0101_0101_0101;
const SWAR_HI: u64 = 0x8080_8080_8080_8080;

impl TokenIndex {
    /// Packs a key of at most `INLINE_MAX` bytes into a word, padded with zeros.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Key to pack.
    #[inline]
    fn inline_word(bytes: &[u8]) -> u64 {
        let mut buf = [0u8; INLINE_MAX];
        buf[..bytes.len()].copy_from_slice(bytes);
        u64::from_le_bytes(buf)
    }

    /// Whether any byte of a zero-padded `word` is [`PAD`].
    ///
    /// XOR maps a padding byte to zero and the SWAR has-zero test finds it. A
    /// zero-padded position holds 0x00, which maps to [`PAD`] and never fires,
    /// so no length is needed.
    ///
    /// # Arguments
    ///
    /// * `word` - Zero-padded key word.
    #[inline]
    fn contains_pad(word: u64) -> bool {
        let x = word ^ PAD_WORD;
        x.wrapping_sub(SWAR_LO) & !x & SWAR_HI != 0
    }

    /// Packs a key of at most `INLINE_MAX` bytes, padding with [`PAD`] so that
    /// the length need not be part of the key. `None` when the key itself
    /// contains [`PAD`] and is therefore not representable.
    ///
    /// The packing is injective over the keys it accepts. Two distinct keys
    /// could only collide if one were the other followed by padding bytes, and
    /// such a key is rejected here, so a padding position can never be confused
    /// with a real byte.
    ///
    /// Rejecting a key costs a fast-path hit but never correctness, because a
    /// miss only sends the caller to the incremental search, which tokenizes the
    /// chunk correctly anyway. In practice the only key any real vocabulary
    /// loses is the single byte [`PAD`] itself.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Key to pack.
    #[inline]
    fn padded_word(bytes: &[u8]) -> Option<u64> {
        let word = Self::inline_word(bytes);
        if Self::contains_pad(word) {
            return None;
        }
        // `checked_shl(64)` is `None`, which yields the empty mask a full-width
        // key needs and avoids the shift-by-64 UB.
        let shift = u32::try_from(8 * bytes.len()).unwrap();
        let mask = (!0u64).checked_shl(shift).unwrap_or(0);
        Some(word | (PAD_WORD & mask))
    }

    /// Builds an index holding every canonical token that is representable.
    ///
    /// # Arguments
    ///
    /// * `table` - The vocabulary under construction.
    pub fn build(table: &TokenTableBuilder) -> Self {
        let mut index = Self {
            short: FxHashMap::default(),
            long: FxHashMap::default(),
            max_len: table.token_len.iter().copied().max().unwrap_or(0),
        };
        index.short.reserve(table.len());
        let k = u32::try_from(table.len()).unwrap();
        for id in 0..k {
            index.insert(table, id);
        }
        index
    }

    /// Places one token in the map matching its length, unless its packed form
    /// would be ambiguous.
    ///
    /// # Arguments
    ///
    /// * `table` - The vocabulary under construction.
    /// * `id` - Internal id of the token to insert.
    fn insert(&mut self, table: &TokenTableBuilder, id: u32) {
        let bytes = table.token_bytes(id);
        let rank = table.rank[usize::from_u32(id)];
        if bytes.len() <= INLINE_MAX {
            if let Some(word) = Self::padded_word(bytes) {
                self.short.insert(word, rank);
            }
        } else {
            self.long.insert(bytes.to_vec(), rank);
        }
    }

    /// Returns the rank of `piece` if it is a canonical token.
    ///
    /// See `docs/measurements.md` for the measurements behind this layout.
    ///
    /// # Arguments
    ///
    /// * `piece` - Chunk to look up.
    #[inline]
    pub fn get(&self, piece: &[u8]) -> Option<u32> {
        // Hashing reads the whole chunk, so reject chunks too long to ever hit.
        if piece.len() > usize::from(self.max_len) {
            return None;
        }
        if piece.len() <= INLINE_MAX {
            return self.short.get(&Self::padded_word(piece)?).copied();
        }
        self.long.get(piece).copied()
    }
}

/// An experimental alternative to [`TokenIndex`]: open addressing over 16-byte
/// slots, with a compact fingerprint array in front to reject misses early.
///
/// Not used by production code. `docs/measurements.md` shows it ahead of [`TokenIndex`] end to
/// end, and further ahead the higher the whole-chunk hit rate gets, so it is worth re-evaluating
/// whenever the pre-tokenization stage changes; [`TokenIndex`] was adopted for being simpler. It
/// owns its token bytes, so keeping it costs the production path nothing.
#[allow(dead_code)]
pub(crate) struct HybridTokenIndex {
    /// One byte per slot, in parallel with `slots`, where 0 marks an empty slot. Compact enough to
    /// stay cache-resident, so a miss usually settles without touching the slot array.
    fingerprints: Vec<u8>,
    slots: Vec<HybridSlot>,
    mask: usize,
    shift: u32,
    max_len: u16,
    /// The byte strings of all tokens, concatenated, for comparing long keys.
    bytes: Vec<u8>,
}

/// One slot of [`HybridTokenIndex`]. `word` is the key itself when
/// `len <= INLINE_MAX`, and an offset into `HybridTokenIndex::bytes` otherwise.
#[allow(dead_code)]
#[derive(Clone, Copy, Default)]
struct HybridSlot {
    word: u64,
    rank: u32,
    len: u16,
}
const _: () = assert!(size_of::<HybridSlot>() == 16);

#[allow(dead_code)]
impl HybridTokenIndex {
    /// Hashes a whole key.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Key to hash.
    #[inline]
    fn hash(bytes: &[u8]) -> u64 {
        let mut hasher = FxHasher::default();
        hasher.write(bytes);
        hasher.finish()
    }

    /// The slot at which a hash value starts probing.
    ///
    /// # Arguments
    ///
    /// * `hash` - Hash value of the key.
    #[inline]
    fn bucket(&self, hash: u64) -> usize {
        usize::try_from(hash >> self.shift).unwrap()
    }

    /// Computes the fingerprint of a hash from its low byte, which the bucket
    /// index (taken from the high bits) does not use. The top bit is forced so
    /// that 0 stays free as the empty marker.
    ///
    /// # Arguments
    ///
    /// * `hash` - Hash value of the key.
    #[inline]
    fn fingerprint(hash: u64) -> u8 {
        u8::try_from(hash & 0xFF).unwrap() | 0x80
    }

    /// Builds an index holding every canonical token.
    ///
    /// # Arguments
    ///
    /// * `table` - The vocabulary under construction.
    pub fn build(table: &TokenTableBuilder) -> Self {
        let cap = (table.len() * 2).next_power_of_two();
        let mut index = Self {
            fingerprints: vec![0u8; cap],
            slots: vec![HybridSlot::default(); cap],
            mask: cap - 1,
            shift: 64 - cap.trailing_zeros(),
            max_len: table.token_len.iter().copied().max().unwrap_or(0),
            bytes: table.bytes.clone(),
        };
        let k = u32::try_from(table.len()).unwrap();
        for id in 0..k {
            index.insert(table, id);
        }
        index
    }

    /// Places one token in an empty slot.
    ///
    /// # Arguments
    ///
    /// * `table` - The vocabulary under construction.
    /// * `id` - Internal id of the token to insert.
    fn insert(&mut self, table: &TokenTableBuilder, id: u32) {
        let bytes = table.token_bytes(id);
        let id = usize::from_u32(id);
        let hash = Self::hash(bytes);
        let mut i = self.bucket(hash);
        while self.fingerprints[i] != 0 {
            i = (i + 1) & self.mask;
        }
        self.fingerprints[i] = Self::fingerprint(hash);
        self.slots[i] = HybridSlot {
            word: if bytes.len() <= INLINE_MAX {
                TokenIndex::inline_word(bytes)
            } else {
                u64::from(table.offsets[id])
            },
            rank: table.rank[id],
            len: table.token_len[id],
        };
    }

    /// Returns the rank of `piece` if it is a canonical token.
    ///
    /// Unlike [`TokenIndex::get`] this has no unrepresentable keys, so it can
    /// answer for chunks containing [`PAD`] as well.
    ///
    /// # Arguments
    ///
    /// * `piece` - Chunk to look up.
    #[inline]
    pub fn get(&self, piece: &[u8]) -> Option<u32> {
        if piece.len() > usize::from(self.max_len) {
            return None;
        }
        let hash = Self::hash(piece);
        let want = Self::fingerprint(hash);
        let mut i = self.bucket(hash);
        if piece.len() <= INLINE_MAX {
            let word = TokenIndex::inline_word(piece);
            let len = u16::try_from(piece.len()).unwrap();
            loop {
                let fingerprint = self.fingerprints[i];
                if fingerprint == 0 {
                    return None;
                }
                if fingerprint == want {
                    let slot = self.slots[i];
                    if slot.len == len && slot.word == word {
                        return Some(slot.rank);
                    }
                }
                i = (i + 1) & self.mask;
            }
        }
        loop {
            let fingerprint = self.fingerprints[i];
            if fingerprint == 0 {
                return None;
            }
            if fingerprint == want {
                let slot = self.slots[i];
                if usize::from(slot.len) == piece.len() {
                    let offset = usize::try_from(slot.word).unwrap();
                    if &self.bytes[offset..offset + piece.len()] == piece {
                        return Some(slot.rank);
                    }
                }
            }
            i = (i + 1) & self.mask;
        }
    }
}

/// Returns the split position of the canonical rule of a non-atomic vocabulary
/// entry, or `None` if the entry is not canonical.
///
/// An entry is canonical when `T_D(t) == [t]`, and the rule producing it is then unique. This runs
/// standard BPE over the token's own byte string and returns the split position of the *last* merge
/// applied, so `Some(mid)` means `pre(t) = t[..mid]` and `suc(t) = t[mid..]`. `None` means the
/// entry is unreachable and may be dropped from V̄.
///
/// Rescanning for the minimum makes this O(|t|²) per token, but it runs once per entry at
/// construction time and |t| is a small constant.
///
/// # Arguments
///
/// * `piece` - Byte string of the token to test.
/// * `ranks` - Map from byte string to rank. A pair absent from the map is
///   treated as "no such rule" (`u32::MAX`).
fn split_canonical(piece: &[u8], ranks: &FxHashMap<&[u8], u32>) -> Option<usize> {
    let n = piece.len();

    // parts[i] = (byte offset of symbol i, priority of the pair (i, i + 1)). The
    // last two are sentinels, so that the pair rank can be read in bounds even
    // for the final symbol.
    let mut parts = Vec::with_capacity(n + 1);
    for i in 0..n - 1 {
        let rank = ranks.get(&piece[i..i + 2]).copied().unwrap_or(u32::MAX);
        parts.push((i, rank));
    }
    parts.push((n - 1, u32::MAX));
    parts.push((n, u32::MAX));

    let mut last_mid = usize::MAX;
    loop {
        let (rank, i) = next_merge(&parts);
        if rank == u32::MAX {
            break;
        }
        last_mid = parts[i + 1].0;
        if i > 0 {
            parts[i - 1].1 = pair_rank(piece, ranks, &parts, i - 1);
        }
        parts[i].1 = pair_rank(piece, ranks, &parts, i);
        parts.remove(i + 1);
    }

    // Only one real element plus the sentinels remain, i.e. the whole string
    // merged into a single token.
    if parts.len() == 2 {
        Some(last_mid)
    } else {
        None
    }
}

/// Returns the priority of the pair starting at `i` *after* `parts[i + 1]` is
/// removed.
///
/// Removing it shifts the far end one position to the left, so in the current
/// array that pair spans `parts[i]..parts[i + 3]`.
///
/// # Arguments
///
/// * `piece` - Byte string of the token being tested.
/// * `ranks` - Map from byte string to rank.
/// * `parts` - The current symbol sequence.
/// * `i` - Left symbol of the pair.
fn pair_rank(piece: &[u8], ranks: &FxHashMap<&[u8], u32>, parts: &[(usize, u32)], i: usize) -> u32 {
    if (i + 3) < parts.len() {
        ranks
            .get(&piece[parts[i].0..parts[i + 3].0])
            .copied()
            .unwrap_or(u32::MAX)
    } else {
        u32::MAX
    }
}

/// Returns the next merge to apply as `(priority, left symbol)`. The priority is
/// `u32::MAX` when no pair can be applied.
///
/// `min_by_key` returns the first of equal keys, so when the same pair occurs
/// twice they merge left to right, as standard BPE requires.
///
/// # Arguments
///
/// * `parts` - The current symbol sequence. The trailing sentinels form no pair
///   and are excluded from the scan.
fn next_merge(parts: &[(usize, u32)]) -> (u32, usize) {
    parts[..parts.len() - 1]
        .iter()
        .enumerate()
        .min_by_key(|&(_, &(_, rank))| rank)
        .map_or((u32::MAX, usize::MAX), |(i, &(_, rank))| (rank, i))
}

/// One element of the normalized vocabulary V̄.
#[derive(Clone, Copy)]
struct CanonicalEntry<'a> {
    /// Byte string of the token.
    bytes: &'a [u8],
    /// The rank, i.e. the position in the original list.
    rank: u32,
    /// Split position of the canonical rule, meaning `pre(t) = t[..mid]` and `suc(t) = t[mid..]`.
    /// `None` exactly for atomic tokens, which have `|t| == 1`.
    mid: Option<usize>,
}

/// Normalizes the vocabulary, returning only the canonical entries with their split positions.
///
/// Non-canonical entries (`T_D(t) != [t]`) are dropped, which loses nothing observable. `tokens` is
/// in ascending rank order, so the result is too, which makes the internal ids follow priority
/// order.
///
/// # Arguments
///
/// * `tokens` - List of token byte strings in priority order.
/// * `ranks` - Map from byte string to rank.
fn filter_canonical<'a, T>(
    tokens: &'a [T],
    ranks: &FxHashMap<&'a [u8], u32>,
) -> Result<Vec<CanonicalEntry<'a>>>
where
    T: AsRef<[u8]>,
{
    let mut canon = Vec::with_capacity(tokens.len());
    for (rank, bytes) in tokens.iter().enumerate() {
        let bytes = bytes.as_ref();
        let rank = u32::try_from(rank).unwrap();
        match bytes.len() {
            0 => return Err(Error::EmptyToken),
            1 => canon.push(CanonicalEntry {
                bytes,
                rank,
                mid: None,
            }),
            _ => {
                if let Some(mid) = split_canonical(bytes, ranks) {
                    canon.push(CanonicalEntry {
                        bytes,
                        rank,
                        mid: Some(mid),
                    });
                }
            }
        }
    }
    Ok(canon)
}

impl TokenTableBuilder {
    /// Validates the vocabulary while building the normalized vocabulary V̄, then linearizes the
    /// Successor Forest.
    ///
    /// Besides normalizing, it checks that every single byte is in the vocabulary so that θ(·) is
    /// always defined, that no byte string appears twice, and that both components outrank the
    /// merged token. That the components of a canonical rule are themselves canonical is not
    /// checked, because it holds by construction. The Successor Forest relies on it.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The ordered list of merge rules. A token's position is its rank.
    ///
    /// # Errors
    ///
    /// See [`Error`] for the conditions a vocabulary is rejected under.
    pub fn new<I, P>(tokens: I) -> Result<Self>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        // A rank is a position, so indexed access is needed, and the byte strings `ranks` borrows
        // need somewhere to live. This single collection covers both.
        let tokens = tokens.into_iter().collect::<Vec<_>>();

        // A rank is a position, so a vocabulary whose length does not fit in a `u32` cannot be
        // represented. This also rules out a collision with the sentinels.
        u32::try_from(tokens.len()).map_err(|_| Error::VocabTooLarge {
            what: "tokens",
            len: tokens.len(),
        })?;

        // Map taking a position as the rank, which the canonicity test looks up.
        let mut ranks = FxHashMap::default();
        ranks.reserve(tokens.len());
        for (rank, bytes) in tokens.iter().enumerate() {
            let bytes = bytes.as_ref();
            let rank = u32::try_from(rank).unwrap();
            if let Some(first) = ranks.insert(bytes, rank) {
                return Err(Error::DuplicateToken {
                    token: bytes.to_vec(),
                    first,
                    second: rank,
                });
            }
        }

        // Every byte must be an atomic token of V̄, since those are the roots of the
        // SufSucTree(τ). An empty vocabulary is rejected here too.
        for b in 0u8..=255 {
            if !ranks.contains_key(&[b][..]) {
                return Err(Error::MissingByteToken(b));
            }
        }

        // The result is in ascending rank order, so the ids assigned below follow priority order.
        let canon = filter_canonical(&tokens, &ranks)?;

        let k = canon.len();
        let mut index = FxHashMap::default();
        index.reserve(k);
        for (i, entry) in canon.iter().enumerate() {
            // The canonical tokens are a subset of the original list, so their
            // count fits in a `u32` too.
            index.insert(entry.bytes, u32::try_from(i).unwrap());
        }

        let total_bytes = canon.iter().map(|entry| entry.bytes.len()).sum();
        let mut table = Self {
            bytes: Vec::with_capacity(total_bytes),
            offsets: Vec::with_capacity(k + 1),
            token_len: Vec::with_capacity(k),
            rank: Vec::with_capacity(k),
            pre: Vec::with_capacity(k),
            suc: Vec::with_capacity(k),
            dfs_in: vec![0; k],
            dfs_out: vec![0; k],
            search: vec![SearchEntry::default(); k],
            dfs_in_inv: vec![DfsEntry::default(); k],
        };
        table.offsets.push(0);

        for &CanonicalEntry { bytes, rank, mid } in &canon {
            // Only tokens longer than `u16::MAX` take this branch, so slicing
            // the first 32 bytes for the message is always in bounds.
            let token_len = u16::try_from(bytes.len()).map_err(|_| Error::TokenTooLong {
                token: bytes[..32].to_vec(),
                len: bytes.len(),
            })?;
            table.bytes.extend_from_slice(bytes);
            // Offsets are `u32`, and a huge vocabulary really can overflow one,
            // so this is an error rather than a silent truncation.
            let end = u32::try_from(table.bytes.len()).map_err(|_| Error::VocabTooLarge {
                what: "vocabulary bytes",
                len: table.bytes.len(),
            })?;
            table.offsets.push(end);
            table.token_len.push(token_len);
            table.rank.push(rank);
            match mid {
                None => {
                    table.pre.push(INVALID_ID);
                    table.suc.push(INVALID_ID);
                }
                Some(mid) => {
                    let pre = &bytes[..mid];
                    let suc = &bytes[mid..];
                    // No earlier merge crosses the split of the final merge, so pre and suc each
                    // merge into a single token, are therefore canonical, and are always in
                    // `index`.
                    let pre_id = *index.get(pre).unwrap();
                    let suc_id = *index.get(suc).unwrap();
                    // In a proper dictionary the components are produced strictly before the rule
                    // that consumes them.
                    let (pre_rank, suc_rank) = (
                        canon[usize::from_u32(pre_id)].rank,
                        canon[usize::from_u32(suc_id)].rank,
                    );
                    if pre_rank >= rank || suc_rank >= rank {
                        return Err(Error::NotProper {
                            token: bytes.to_vec(),
                            rank,
                            pre: pre.to_vec(),
                            pre_rank,
                            suc: suc.to_vec(),
                            suc_rank,
                        });
                    }
                    table.pre.push(pre_id);
                    table.suc.push(suc_id);
                }
            }
        }

        // pre/suc are in place, so linearize the Successor Forest.
        table.build_forest();
        Ok(table)
    }

    /// Hands over the arrays tokenization needs and drops the ones that were
    /// only used during construction.
    pub fn finish(self) -> TokenTable {
        TokenTable {
            dfs_in: self.dfs_in,
            dfs_in_inv: self.dfs_in_inv,
        }
    }

    /// The number of canonical tokens |V̄|.
    #[inline]
    pub fn len(&self) -> usize {
        self.token_len.len()
    }

    /// The byte string of token `i`.
    ///
    /// # Arguments
    ///
    /// * `i` - Internal token id.
    #[inline]
    pub fn token_bytes(&self, i: u32) -> &[u8] {
        let i = usize::from_u32(i);
        let start = usize::from_u32(self.offsets[i]);
        let end = usize::from_u32(self.offsets[i + 1]);
        &self.bytes[start..end]
    }

    /// Whether `i` is an atomic token, i.e. a root of the Successor Forest.
    ///
    /// An atomic token has no canonical rule and therefore no pre(·).
    ///
    /// # Arguments
    ///
    /// * `i` - Internal token id.
    #[inline]
    pub fn is_atomic(&self, i: u32) -> bool {
        self.pre[usize::from_u32(i)] == INVALID_ID
    }

    /// Builds the automaton payload to attach to the pattern for token `t`.
    ///
    /// # Arguments
    ///
    /// * `t` - Internal token id.
    pub fn to_entry(&self, t: u32) -> StateEntry {
        let t = usize::from_u32(t);
        let search = self.search[t];
        StateEntry {
            valid: search.valid,
            dfs_in_tau: self.dfs_in[t],
            len: search.len,
            suc_len: search.suc_len,
        }
    }

    /// Linearizes the Successor Forest with a DFS over the pre/suc edges, filling `dfs_in`,
    /// `dfs_out`, `dfs_in_inv` and `search`.
    ///
    /// The forest consists of the edges from each non-atomic token to its successor token. Since
    /// `|suc(u)| < |u|` it is acyclic and its roots are exactly the atomic tokens. It is not kept
    /// and exists only for the three passes below.
    fn build_forest(&mut self) {
        let k = self.len();
        let k32 = u32::try_from(k).unwrap();

        // Pass 1. Bucket every non-atomic token under its successor token. The
        // child lists are held in CSR form built by counting sort rather than as
        // a Vec of Vecs, so the children of `u` are
        // `list[start[u]..start[u + 1]]`.
        let mut start = vec![0u32; k + 1];
        for u in 0..k {
            if self.pre[u] != INVALID_ID {
                start[usize::from_u32(self.suc[u]) + 1] += 1;
            }
        }
        for i in 0..k {
            start[i + 1] += start[i];
        }
        let mut cursor = start.clone();
        let mut list = vec![0u32; usize::from_u32(start[k])];
        for u in 0..k32 {
            let ui = usize::from_u32(u);
            if self.pre[ui] != INVALID_ID {
                let p = usize::from_u32(self.suc[ui]);
                list[usize::from_u32(cursor[p])] = u;
                cursor[p] += 1;
            }
        }
        // Children are visited in strictly ascending order of rule priority, which gathers the
        // ones a valid interval must exclude at the end of the parent's range so that a single
        // upper bound cuts them off. Lower priority means larger rank, hence descending rank.
        for u in 0..k {
            let siblings = usize::from_u32(start[u])..usize::from_u32(start[u + 1]);
            list[siblings].sort_unstable_by(|&a, &b| {
                self.rank[usize::from_u32(b)].cmp(&self.rank[usize::from_u32(a)])
            });
        }

        // Pass 2. Run a preorder DFS from each atomic-token root, assigning timestamps so that the
        // subtree of `u` is the half-open interval [dfs_in(u), dfs_out(u)). The depth of a
        // successor-token chain is attacker-influenced, so this uses an explicit stack.
        let mut counter = 0;
        let mut stack = Vec::with_capacity(64);
        for root in 0..k32 {
            if !self.is_atomic(root) {
                continue;
            }
            self.dfs_in[usize::from_u32(root)] = counter;
            counter += 1;
            stack.push((root, start[usize::from_u32(root)]));
            while let Some(&(node, ptr)) = stack.last() {
                if ptr < start[usize::from_u32(node) + 1] {
                    stack.last_mut().unwrap().1 = ptr + 1;
                    let child = list[usize::from_u32(ptr)];
                    self.dfs_in[usize::from_u32(child)] = counter;
                    counter += 1;
                    stack.push((child, start[usize::from_u32(child)]));
                } else {
                    self.dfs_out[usize::from_u32(node)] = counter;
                    stack.pop();
                }
            }
        }

        // Invert the bijection for the backtracking.
        for t in 0..k {
            self.dfs_in_inv[usize::from_u32(self.dfs_in[t])] = DfsEntry {
                rank: self.rank[t],
                len: self.token_len[t],
            };
        }

        // Pass 3. The valid interval I_t = [L_t, R_t).
        for t in 0..k {
            self.search[t].len = self.token_len[t];
            // An atomic token satisfies the condition unconditionally, so its interval is never
            // read. Leaving `suc_len` at 0 is how `descend` recognizes one.
            if self.pre[t] == INVALID_ID {
                continue;
            }
            let p = usize::from_u32(self.pre[t]);
            let rank_t = self.rank[t];
            let siblings = &list[usize::from_u32(start[p])..usize::from_u32(start[p + 1])];
            // R_t is the dfs_in of the first child of pre(t) whose priority is
            // at least t's, or dfs_out(pre(t)) if there is none. The siblings are
            // in descending rank order, so this is a binary search.
            let idx = siblings.partition_point(|&c| self.rank[usize::from_u32(c)] > rank_t);
            // L_t = dfs_in(pre(t)) accounts for reachability, i.e. θ(s−suc(t))
            // lying in the subtree of pre(t). The upper bound is priority dominance, and by
            // the DFS visit order the children being cut off are contiguous at
            // the end of the range.
            let l = self.dfs_in[p];
            let r = if idx < siblings.len() {
                self.dfs_in[usize::from_u32(siblings[idx])]
            } else {
                self.dfs_out[p]
            };
            let suc_len = self.token_len[usize::from_u32(self.suc[t])];
            self.search[t].valid = (l..r).into();
            self.search[t].suc_len = suc_len;
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The xorshift generator the equivalence test uses.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            usize::try_from(self.next() % u64::try_from(n).unwrap()).unwrap()
        }
    }

    /// A small proper vocabulary consisting of all 256 single bytes plus random
    /// merges over a tiny alphabet that deliberately includes [`PAD`], so that both the
    /// representable and the unrepresentable key paths are exercised.
    ///
    /// # Arguments
    ///
    /// * `seed` - Seed for the merge generator, so the sweep is deterministic.
    pub(crate) fn sample_vocabulary(seed: u64) -> Vec<Vec<u8>> {
        let alphabet = [b'a', b'b', PAD];
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        let mut vocab: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let mut grown: Vec<Vec<u8>> = alphabet.iter().map(|&b| vec![b]).collect();
        for _ in 0..40 {
            let x = grown[rng.below(grown.len())].clone();
            let y = grown[rng.below(grown.len())].clone();
            let mut merged = x;
            merged.extend_from_slice(&y);
            if merged.len() > 12 || vocab.contains(&merged) {
                continue;
            }
            vocab.push(merged.clone());
            grown.push(merged);
        }
        // Growing merges does not by itself keep the dictionary proper, so drop
        // the entries the builder rejects and retry.
        loop {
            match TokenTableBuilder::new(&vocab) {
                Ok(_) => return vocab,
                Err(Error::NotProper { token, .. }) => vocab.retain(|t| *t != token),
                Err(e) => panic!("unexpected build error: {e}"),
            }
        }
    }

    fn sample_table(seed: u64) -> TokenTableBuilder {
        TokenTableBuilder::new(sample_vocabulary(seed)).unwrap()
    }

    /// How often the two indexes agreed, and how often they diverged on a key
    /// [`TokenIndex`] cannot represent.
    struct Coverage {
        hits: usize,
        pad_misses: usize,
    }

    /// Compares the two indexes on one query and folds the outcome into
    /// `coverage`, failing if they disagree in a way the design does not allow.
    ///
    /// # Arguments
    ///
    /// * `index` - The production index.
    /// * `hybrid` - The experimental index.
    /// * `query` - The chunk to look up in both.
    /// * `coverage` - Running outcome counts.
    fn compare_one(
        index: &TokenIndex,
        hybrid: &HybridTokenIndex,
        query: &[u8],
        coverage: &mut Coverage,
    ) {
        let short_with_pad = query.len() <= INLINE_MAX && query.contains(&PAD);
        match (index.get(query), hybrid.get(query)) {
            (Some(a), Some(b)) => {
                assert_eq!(a, b, "ranks disagree for {query:?}");
                coverage.hits += 1;
            }
            (None, Some(_)) => {
                assert!(short_with_pad);
                coverage.pad_misses += 1;
            }
            (Some(_), None) => panic!("TokenIndex found {query:?} but the hybrid did not"),
            // A genuine miss on both sides needs no further check.
            (None, None) => {}
        }
    }

    /// Every token of `table`, then random strings over `alphabet` that mostly
    /// miss.
    ///
    /// # Arguments
    ///
    /// * `table` - The vocabulary whose tokens seed the query list.
    /// * `seed` - Seed for the random strings, so the sweep is deterministic.
    fn sample_queries(table: &TokenTableBuilder, seed: u64) -> Vec<Vec<u8>> {
        let alphabet = [b'a', b'b', PAD];
        let mut queries: Vec<Vec<u8>> = (0..u32::try_from(table.len()).unwrap())
            .map(|id| table.token_bytes(id).to_vec())
            .collect();
        let mut rng = Rng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1);
        for _ in 0..2000 {
            let len = 1 + rng.below(14);
            queries.push((0..len).map(|_| alphabet[rng.below(3)]).collect());
        }
        queries
    }

    /// Runs the comparison over `seeds` vocabularies and returns what it covered.
    ///
    /// # Arguments
    ///
    /// * `seeds` - How many vocabularies to sweep.
    fn compare_indexes(seeds: u64) -> Coverage {
        let mut coverage = Coverage {
            hits: 0,
            pad_misses: 0,
        };
        for seed in 1..=seeds {
            let table = sample_table(seed);
            let index = TokenIndex::build(&table);
            let hybrid = HybridTokenIndex::build(&table);
            for query in sample_queries(&table, seed) {
                compare_one(&index, &hybrid, &query, &mut coverage);
            }
        }
        coverage
    }

    /// [`HybridTokenIndex`] must answer exactly as [`TokenIndex`] does, except
    /// that [`TokenIndex`] deliberately misses short keys containing [`PAD`].
    #[test]
    fn hybrid_index_agrees_with_token_index() {
        let coverage = compare_indexes(8);

        // The sweep must actually have covered both outcomes.
        assert!(coverage.hits > 0);
        assert!(coverage.pad_misses > 0);
    }

    /// All 256 single-byte tokens, which every vocabulary must contain.
    fn single_bytes() -> Vec<Vec<u8>> {
        (0u8..=255).map(|b| vec![b]).collect()
    }

    /// A deterministic vocabulary whose merges grow by doubling, so that it
    /// holds a key on each side of the `INLINE_MAX` boundary.
    fn doubling_vocabulary() -> Vec<Vec<u8>> {
        let mut vocab = single_bytes();
        vocab.push(b"ab".to_vec());
        vocab.push(b"abab".to_vec());
        vocab.push(b"abababab".to_vec());
        vocab.push(b"ababababab".to_vec());
        vocab
    }

    /// A vocabulary holding both `[0x00]` and `[0x00, 0x00]`, the pair that zero
    /// padding alone would map to the same word.
    fn zero_pair_vocabulary() -> Vec<Vec<u8>> {
        let mut vocab = single_bytes();
        vocab.push(vec![0x00, 0x00]);
        vocab
    }

    /// Builds the index for `vocab`, which must be proper.
    ///
    /// # Arguments
    ///
    /// * `vocab` - The token list to compile.
    fn index_of(vocab: &[Vec<u8>]) -> TokenIndex {
        let table = TokenTableBuilder::new(vocab).unwrap();
        TokenIndex::build(&table)
    }

    /// Asserts that every probe holds [`PAD`] and is therefore unrepresentable.
    ///
    /// # Arguments
    ///
    /// * `probes` - Keys that must all contain the padding byte.
    fn assert_all_hold_pad(probes: &[Vec<u8>]) {
        for probe in probes {
            assert!(TokenIndex::contains_pad(TokenIndex::inline_word(probe)));
            assert_eq!(TokenIndex::padded_word(probe), None);
        }
    }

    /// Asserts that no probe holds [`PAD`], so all of them are representable.
    ///
    /// # Arguments
    ///
    /// * `probes` - Keys that must all be free of the padding byte.
    fn assert_none_hold_pad(probes: &[Vec<u8>]) {
        for probe in probes {
            assert!(!TokenIndex::contains_pad(TokenIndex::inline_word(probe)));
            assert!(TokenIndex::padded_word(probe).is_some());
        }
    }

    /// Exercises [`TokenIndex::get`] over every path, namely a short hit, a hit exactly at the
    /// `INLINE_MAX` boundary, a long hit, a short miss, a chunk past `max_len`,
    /// and the deliberate miss on a key containing [`PAD`].
    #[test]
    fn token_index_get_covers_every_path() {
        let index = index_of(&doubling_vocabulary());

        assert_eq!(index.get(b"a"), Some(u32::from(b'a')));
        assert_eq!(index.get(b"ab"), Some(256));
        // Exactly `INLINE_MAX` bytes, so the packing mask is empty.
        assert_eq!(index.get(b"abababab"), Some(258));
        // Longer than `INLINE_MAX`, so it lives in the owned-key map.
        assert_eq!(index.get(b"ababababab"), Some(259));

        assert_eq!(index.get(b"ba"), None);
        assert_eq!(index.get(b"ababab"), None);
        // `max_len` is 10 here, so anything longer is rejected before hashing.
        assert_eq!(index.get(b"abababababab"), None);

        // `[PAD]` is a token, but it is not representable, so the index misses
        // it on purpose and the caller falls back to the incremental search.
        assert_eq!(index.get(&[PAD]), None);
    }

    /// The padding is what makes the packing injective. Zero padding alone maps
    /// `[0x00]` and `[0x00, 0x00]` to the same word, and both are in every real
    /// vocabulary.
    #[test]
    fn token_index_distinguishes_keys_that_zero_padding_would_merge() {
        let index = index_of(&zero_pair_vocabulary());

        assert_eq!(index.get(&[0x00]), Some(0));
        assert_eq!(index.get(&[0x00, 0x00]), Some(256));
    }

    /// [`TokenIndex::contains_pad`] must see [`PAD`] anywhere in the real bytes
    /// and must never fire on the zero padding.
    #[test]
    fn contains_pad_finds_the_padding_byte_only_in_real_bytes() {
        assert_all_hold_pad(&[
            vec![PAD],
            vec![PAD, b'a'],
            vec![b'a', PAD],
            vec![b'a', b'b', b'c', b'd', b'e', b'f', b'g', PAD],
            vec![0x00, PAD],
        ]);
        assert_none_hold_pad(&[
            vec![b'a'],
            vec![0x00],
            vec![0x00, 0x00],
            vec![0xF4],
            vec![0xF6],
            vec![b'a'; INLINE_MAX],
        ]);
    }
}
