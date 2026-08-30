//! Utility functions.
//!
//! Table indices are held as `u32` and widened to `usize` on every use. `as` can silently corrupt a
//! value, so this module offers only conversions that cannot fail on a 32-bit or 64-bit target.

/// A type that can be converted from a `u32` without narrowing.
pub trait FromU32 {
    /// Converts `src`.
    ///
    /// # Arguments
    ///
    /// * `src` - Value to convert.
    fn from_u32(src: u32) -> Self;
}

#[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
impl FromU32 for usize {
    #[inline(always)]
    fn from_u32(src: u32) -> Self {
        // The pointer width is 32 or 64, so this always succeeds.
        Self::try_from(src).unwrap()
    }
}
