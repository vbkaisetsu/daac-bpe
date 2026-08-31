//! Construction of [`IncrementalBpe`], i.e. the vocabulary preprocessing.

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder};

use crate::sufsuc::CentroidSearchTree;
use crate::utils::FromU32;
use crate::vocab::{FxHashMap, StateEntry, TokenIndex, TokenTableBuilder};
use crate::{Error, IncrementalBpe, Result};

/// Incremental BPE builder.
#[derive(Clone, Debug, Default)]
pub struct IncrementalBpeBuilder {
    corpus: Vec<Vec<u8>>,
    specials: Vec<(Vec<u8>, u32)>,
}

impl IncrementalBpeBuilder {
    /// Creates a new builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use daac_bpe::IncrementalBpeBuilder;
    ///
    /// let patterns = (0u8..=255).map(|b| [b]);
    ///
    /// let bpe = IncrementalBpeBuilder::new().build(patterns)?;
    /// # Ok::<(), daac_bpe::Error>(())
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self {
            corpus: vec![],
            specials: vec![],
        }
    }

    /// Specifies a corpus of sample chunks for the profile-guided layout optimization.
    ///
    /// # Arguments
    ///
    /// * `haystacks` - List of chunks. Each element corresponds to a chunk passed to
    ///   [`IncrementalBpe::encode()`].
    ///
    /// # Examples
    ///
    /// ```
    /// use daac_bpe::IncrementalBpeBuilder;
    ///
    /// let patterns = (0u8..=255).map(|b| [b]);
    ///
    /// let bpe = IncrementalBpeBuilder::new()
    ///     .corpus(["a sample", " of", " the", " text"])
    ///     .build(patterns)?;
    /// # Ok::<(), daac_bpe::Error>(())
    /// ```
    #[must_use]
    pub fn corpus<I, P>(mut self, corpus: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        self.corpus = corpus.into_iter().map(|h| h.as_ref().to_vec()).collect();
        self
    }

    /// Registers special tokens, which the same automaton detects while it drives the BPE.
    ///
    /// An occurrence finalizes the BPE of the text before it, emits its own id, and restarts the
    /// tokenization behind it, so no vocabulary token spans one of its boundaries. The ids must lie
    /// outside the vocabulary's rank space, i.e. be at least the length of the token list passed to
    /// [`build`](Self::build).
    ///
    /// No canonical vocabulary token and no other special token must not contain a special token as
    /// a substring.
    ///
    /// # Arguments
    ///
    /// * `specials` - List of `(byte string, id)` pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use daac_bpe::IncrementalBpeBuilder;
    ///
    /// let patterns = (0u8..=255).map(|b| [b]);
    ///
    /// let bpe = IncrementalBpeBuilder::new()
    ///     .special_tokens([(b"<|endoftext|>", 256)])
    ///     .build(patterns)?;
    /// let mut tokens = vec![];
    /// bpe.encode(b"a<|endoftext|>b", &mut tokens);
    /// assert_eq!(tokens, [u32::from(b'a'), 256, u32::from(b'b')]);
    /// # Ok::<(), daac_bpe::Error>(())
    /// ```
    #[must_use]
    pub fn special_tokens<I, P>(mut self, specials: I) -> Self
    where
        I: IntoIterator<Item = (P, u32)>,
        P: AsRef<[u8]>,
    {
        self.specials = specials
            .into_iter()
            .map(|(bytes, id)| (bytes.as_ref().to_vec(), id))
            .collect();
        self
    }

    /// Builds a tokenizer.
    ///
    /// The vocabulary must be *proper*: for every canonical rule, the ranks of both components must
    /// be strictly smaller than that of the merged token. Non-canonical entries (`T_D(t) != [t]`)
    /// are dropped silently.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The ordered list of merge rules. A token's position is its rank.
    ///
    /// # Examples
    ///
    /// ```
    /// use daac_bpe::IncrementalBpeBuilder;
    ///
    /// let patterns = (0u8..=255).map(|b| [b]);
    ///
    /// let bpe = IncrementalBpeBuilder::new().build(patterns)?;
    /// let mut tokens = vec![];
    /// bpe.encode(b"hi", &mut tokens);
    /// assert_eq!(tokens, [u32::from(b'h'), u32::from(b'i')]);
    /// # Ok::<(), daac_bpe::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// * [`Error::EmptyToken`] - A token or a special token of zero length is present.
    /// * [`Error::DuplicateToken`] - The same byte string appears twice, in the vocabulary or among
    ///   the special tokens.
    /// * [`Error::VocabTooLarge`] - The token count or the total byte length does not fit in a
    ///   `u32`.
    /// * [`Error::MissingByteToken`] - A single-byte token is missing.
    /// * [`Error::NotProper`] - The dictionary is not proper.
    /// * [`Error::TokenTooLong`] - A token or a special token longer than `u16::MAX` is present.
    /// * [`Error::SpecialIdConflict`] - A special token id lies in the vocabulary's rank space or
    ///   is claimed by another special token.
    /// * [`Error::SpecialTokenNotIsolated`] - A special token occurs inside a canonical token or
    ///   inside another special token.
    /// * [`Error::Automaton`] - Building the automaton failed.
    pub fn build<I, P>(self, tokens: I) -> Result<IncrementalBpe>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        // A rank is a position, so the rank space is the *original* list length, read before
        // `TokenTableBuilder` drops the non-canonical entries.
        let tokens = tokens.into_iter().collect::<Vec<_>>();
        let rank_space = u32::try_from(tokens.len()).map_err(|_| Error::VocabTooLarge {
            what: "tokens",
            len: tokens.len(),
        })?;

        let tables = TokenTableBuilder::new(&tokens)?;
        // Must precede the automaton: `CentroidSearchTree::build` reads `dfs_in_tau` of every match
        // it sees, and only the substring constraint keeps a special token out of those matches.
        validate_specials(&self.specials, &tables, rank_space)?;

        let k = u32::try_from(tables.len()).unwrap();
        // `build_with_values` rejects duplicate patterns, which the validation above rules out.
        let patterns = (0..k)
            .map(|i| (tables.token_bytes(i).to_vec(), tables.to_entry(i)))
            .chain(self.specials.iter().map(|(bytes, id)| {
                let len = u16::try_from(bytes.len()).unwrap();
                (bytes.clone(), StateEntry::special(*id, len))
            }));
        let pma = DoubleArrayAhoCorasickBuilder::new()
            .corpus(&self.corpus)
            .build_with_values::<_, _, StateEntry>(patterns)
            .map_err(|e| Error::Automaton(e.to_string()))?;

        let centroid_trees = CentroidSearchTree::build(&tables, &pma);
        let index = TokenIndex::build(&tables, &self.specials);

        let table = tables.finish();
        let mut bpe = IncrementalBpe {
            table,
            pma,
            centroid_trees,
            index,
        };
        if !self.corpus.is_empty() {
            let counts = bpe.count_cst_visits(&self.corpus);
            bpe.centroid_trees.reorder_by_counts(&counts);
        }
        Ok(bpe)
    }
}

/// Checks everything the runtime assumes about the special tokens.
///
/// The per-entry checks come first, so that the automaton of the substring test only ever sees
/// non-empty patterns.
///
/// # Arguments
///
/// * `specials` - The registered `(byte string, id)` pairs, in registration order.
/// * `tables` - The normalized vocabulary, whose canonical tokens are scanned for occurrences.
/// * `rank_space` - Length of the original token list, i.e. the smallest usable special token id.
///
/// # Errors
///
/// See [`IncrementalBpeBuilder::build`] for the conditions the special tokens are rejected under.
fn validate_specials(
    specials: &[(Vec<u8>, u32)],
    tables: &TokenTableBuilder,
    rank_space: u32,
) -> Result<()> {
    if specials.is_empty() {
        return Ok(());
    }

    // Both maps record the position that first claimed a byte string or an id.
    let mut seen_bytes = FxHashMap::default();
    let mut seen_ids = FxHashMap::default();
    seen_bytes.reserve(specials.len());
    seen_ids.reserve(specials.len());
    for (pos, (bytes, id)) in specials.iter().enumerate() {
        // The ids are distinct `u32` values, so a list long enough to overflow this conversion
        // would have raised a conflict at an earlier position.
        let pos = u32::try_from(pos).unwrap();
        if bytes.is_empty() {
            return Err(Error::EmptyToken);
        }
        // Only a special token longer than `u16::MAX` takes this branch, so slicing the first 32
        // bytes for the message is always in bounds.
        u16::try_from(bytes.len()).map_err(|_| Error::TokenTooLong {
            token: bytes[..32].to_vec(),
            len: bytes.len(),
        })?;
        // Every value below the rank space is a rank, taken whether its entry survived or not.
        if *id < rank_space || seen_ids.insert(*id, pos).is_some() {
            return Err(Error::SpecialIdConflict {
                token: bytes.clone(),
                id: *id,
            });
        }
        if let Some(first) = seen_bytes.insert(bytes.as_slice(), pos) {
            return Err(Error::DuplicateToken {
                token: bytes.clone(),
                first,
                second: pos,
            });
        }
    }

    // A throwaway automaton over the special tokens alone, valued by their positions.
    // `find_overlapping_iter` reports the suffix matches too, so one scan sees every occurrence.
    let probe = DoubleArrayAhoCorasick::<u32>::new(specials.iter().map(|(bytes, _)| bytes))
        .map_err(|e| Error::Automaton(e.to_string()))?;

    let k = u32::try_from(tables.len()).unwrap();
    for i in 0..k {
        let token = tables.token_bytes(i);
        if let Some(m) = probe.find_overlapping_iter(token).next() {
            return Err(Error::SpecialTokenNotIsolated {
                special: specials[usize::from_u32(m.value())].0.clone(),
                container: token.to_vec(),
            });
        }
    }
    for (bytes, _) in specials {
        for m in probe.find_overlapping_iter(bytes) {
            // The byte strings are distinct by now, so a match spanning the whole haystack is the
            // token's own identity match.
            if m.start() == 0 && m.end() == bytes.len() {
                continue;
            }
            return Err(Error::SpecialTokenNotIsolated {
                special: specials[usize::from_u32(m.value())].0.clone(),
                container: bytes.clone(),
            });
        }
    }
    Ok(())
}
