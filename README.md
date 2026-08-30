# daac-bpe

A BPE tokenizer that operates in linear time relative to the input length.

While this implementation is based on the following paper, various optimizations have been
applied to improve performance.

> Shenghu Jiang and Ruihao Gong. *Incremental BPE Tokenization.*
> ICML 2026. <https://arxiv.org/abs/2605.30813>

## Examples

A vocabulary is an ordered list of token byte strings, and a token's rank is its position in the
list. All 256 single-byte tokens must be present; merge rules follow in priority order.

```rust
use daac_bpe::IncrementalBpe;

let mut vocab: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
vocab.push(b"ab".to_vec()); // rank 256
vocab.push(b"abc".to_vec()); // rank 257

let bpe = IncrementalBpe::new(&vocab)?;

// Tokens are appended, so one buffer can be reused across chunks.
let mut tokens = vec![];
bpe.encode(b"abcab", &mut tokens);
assert_eq!(tokens, [257, 256]);
```

To optimize the memory layout for a sample of the text to be tokenized, build the tokenizer with
`IncrementalBpeBuilder::corpus()`.

## Benchmark

One sample is one whole-corpus pass; implementations are interleaved within each
round (forward then reverse order) so CPU frequency drift cancels, pinned to one
P-core of an i7-1270P. All output and working buffers are allocated once and
reused across passes. Full tables, methodology, and reproduction commands:
[`bench/RESULTS-abba-o200k.md`](bench/RESULTS-abba-o200k.md).

### Tokenizers

Pure incremental BPE tokenizers (no pre-tokenization stage and directly comparable):

* `daac-bpe`: this crate
* [`mtc-inc-bpe`](https://github.com/ModelTC/mtc-inc-bpe) 0.9.2: the paper authors' implementation
* [`bpe`](https://github.com/github/rust-gems) 0.2.2 (rust-gems)

BPE tokenizers with pre-tokenization (end-to-end, including their own regex stage. Reference records
only, not directly comparable with the group above):

* [`tiktoken`](https://github.com/openai/tiktoken) (0.14.0, Rust core)
* [`tokenizers`](https://github.com/huggingface/tokenizers) 0.23
* [`fastokens`](https://crates.io/crates/fastokens) 0.3

### Dataset

* en_wiki — English Wikipedia (20231101, stride-sampled), 2.05 MB
* ja_wiki — Japanese Wikipedia (same sampling), 2.05 MB
* code — 10 large OSS source files, 1.57 MB

### Token set

* o200k_base (<https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken>)

### Results

Median MiB/s over 22 passes, newline-split chunks. Higher is faster.

| tokenizer         | en_wiki  | ja_wiki  | code     |
|-------------------|---------:|---------:|---------:|
| **`daac-bpe`**    | **16.4** | **34.4** | **20.2** |
| `mtc-inc-bpe`     |     12.0 |     30.0 |     16.1 |
| `bpe` (rust-gems) |     10.4 |     13.9 |     13.1 |

Reference records (pre-tokenization included):

| tokenizer    | en_wiki | ja_wiki | code |
|--------------|--------:|--------:|-----:|
| `tiktoken`   |    10.2 |     5.6 |  9.1 |
| `tokenizers` |     1.5 |     2.0 |  1.3 |
| `fastokens`  |    69.7 |     9.4 | 42.5 |

### Caveats

* The two groups measure different pipelines and must not be compared with each other. `fastokens`'
  lead on en_wiki/code comes largely from its PCRE2+JIT pre-tokenizer and internal caches (warm in
  this benchmark), not its BPE stage.
* `bpe` (rust-gems) allocates its working buffers per chunk through its public API, while the
  other two reuse caller-owned buffers. For a fair comparison it was measured through a local,
  purely additive patch that reuses those buffers across chunks (output verified identical to the
  unpatched crate).

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Notes

Most of this implementation was written with coding agents, but all of the code was then reviewed,
verified, and corrected by hand.
