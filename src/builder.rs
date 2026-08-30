//! Construction of [`IncrementalBpe`], i.e. the vocabulary preprocessing.

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use daachorse::DoubleArrayAhoCorasickBuilder;

use crate::sufsuc::CentroidSearchTree;
use crate::vocab::{StateEntry, TokenIndex, TokenTableBuilder};
use crate::{Error, IncrementalBpe, Result};

/// Incremental BPE builder.
#[derive(Clone, Debug, Default)]
pub struct IncrementalBpeBuilder {
    corpus: Vec<Vec<u8>>,
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
        Self { corpus: vec![] }
    }

    /// Specifies a corpus of sample chunks for the profile-guided layout optimization.
    ///
    /// # Arguments
    ///
    /// * `haystacks` - List of chunks. Each element corresponds to a chunk
    ///   passed to [`IncrementalBpe::encode()`].
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
    /// * [`Error::EmptyToken`] - A token of zero-length is present.
    /// * [`Error::DuplicateToken`] - The same byte string appears twice.
    /// * [`Error::VocabTooLarge`] - The token count or the total byte length does not fit in a `u32`.
    /// * [`Error::MissingByteToken`] - A single-byte token is missing.
    /// * [`Error::NotProper`] - The dictionary is not proper.
    /// * [`Error::TokenTooLong`] - A token longer than `u16::MAX` is present.
    /// * [`Error::Automaton`] - Building the automaton failed.
    pub fn build<I, P>(self, tokens: I) -> Result<IncrementalBpe>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        let tables = TokenTableBuilder::new(tokens)?;
        let k = u32::try_from(tables.len()).unwrap();
        let patterns = (0..k).map(|i| (tables.token_bytes(i).to_vec(), tables.to_entry(i)));
        let pma = DoubleArrayAhoCorasickBuilder::new()
            .corpus(&self.corpus)
            .build_with_values::<_, _, StateEntry>(patterns)
            .map_err(|e| Error::Automaton(e.to_string()))?;

        let centroid_trees = CentroidSearchTree::build(&tables, &pma);
        let index = TokenIndex::build(&tables);

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
