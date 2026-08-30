//! Errors of this crate.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// Error variants of this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An entry of length zero is present.
    #[error("The vocabulary contains an empty token")]
    EmptyToken,

    /// The same byte string appears twice.
    #[error("The token {} appears twice, at positions {first} and {second}", Bytes(.token))]
    DuplicateToken {
        token: Vec<u8>,
        first: u32,
        second: u32,
    },

    /// The vocabulary is too large to index with a `u32`.
    #[error("the vocabulary is too large: {len} {what} do not fit in a u32 index")]
    VocabTooLarge { what: &'static str, len: usize },

    /// Some byte value has no corresponding atomic token.
    #[error("The single-byte token 0x{0:02x} is missing")]
    MissingByteToken(u8),

    /// The dictionary is *not proper*, i.e. a component of some canonical rule does not come
    /// before the merged token, which would require standard BPE to return to a priority it has
    /// already passed.
    #[error(
        "vocabulary is not proper: the canonical rule ({} [rank {pre_rank}], {} [rank {suc_rank}]) \
        -> {} [rank {rank}] has a component whose rank is not smaller than the merged token's rank",
        Bytes(.pre),
        Bytes(.suc),
        Bytes(.token)
    )]
    NotProper {
        token: Vec<u8>,
        rank: u32,
        pre: Vec<u8>,
        pre_rank: u32,
        suc: Vec<u8>,
        suc_rank: u32,
    },

    /// A token longer than `u16::MAX` is present, which the search entries cannot hold.
    #[error(
        "the token starting with {} is {len} bytes long; the limit is {}",
        Bytes(.token),
        u16::MAX
    )]
    TokenTooLong { token: Vec<u8>, len: usize },

    /// Building the Aho-Corasick automaton failed.
    #[error("failed to build the Aho-Corasick automaton: {0}")]
    Automaton(String),
}

/// Specialized result type of this crate.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// Formats a token's byte string for a message, quoted if it is printable ASCII and as a `0x…`
/// hex dump otherwise. A `Display` adapter rather than a helper, so nothing is allocated.
struct Bytes<'a>(&'a [u8]);

impl fmt::Display for Bytes<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.iter().all(|&b| b == b' ' || b.is_ascii_graphic()) {
            write!(f, "{:?}", String::from_utf8_lossy(self.0))
        } else {
            f.write_str("0x")?;
            self.0.iter().try_for_each(|b| write!(f, "{b:02x}"))
        }
    }
}
