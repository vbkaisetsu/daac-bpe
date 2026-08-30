//! The suffix-successor tree and its Centroid Search Tree.
//!
//! SufSucTree(τ) is the subgraph of the Successor Forest induced by the canonical tokens that are
//! suffixes of τ. They all share their last byte, so it is a single tree rooted at the atomic
//! suffix token of τ. That tree is the search space for θ(sc), but its height can be proportional
//! to |τ|, so this precomputes a *Centroid Search Tree*, which splits it recursively at its
//! centroids so that a search finishes in O(log |τ|) steps.

use alloc::vec;
use alloc::vec::Vec;
use core::range::Range;

use daachorse::DoubleArrayAhoCorasick;

use crate::utils::FromU32;
use crate::vocab::{INVALID_ID, SearchEntry, StateEntry, TokenTableBuilder};

/// One centroid in the decomposition of some SufSucTree(τ).
///
/// Removing it splits the tree into one component holding the parent, which is `up_component`, and
/// one per child, which are the `child_len` elements starting at `child_start`.
#[derive(Clone, Copy)]
pub(crate) struct CentroidSearchTreeNode {
    /// The token this centroid stands for. If the search stops here, that is the answer θ(sc).
    pub token: u32,
    /// The CST node for the component holding `token`'s parent, or `INVALID_ID` when the parent is
    /// no longer in the component. The search never reads that value, because on entering a
    /// component it already knows the answer lies below its topmost node.
    pub up_component: u32,
    /// Start of this node's slice of `CentroidSearchTree::children`.
    pub child_start: u32,
    /// The number of children of `token`, counting *all* of them rather than only those left in
    /// the component, so that the binary search sees the original sibling set.
    pub child_len: u32,
    /// A copy of `table.search[token]`, so that one descent step reads only this node.
    pub search: SearchEntry,
}

/// An edge from a centroid to one of its children. Carrying the interval lets the search binary
/// search without touching the token table.
#[derive(Clone, Copy)]
pub(crate) struct CentroidSearchTreeChild {
    /// The child's valid interval I_v. Sibling intervals are disjoint, so they can be sorted and
    /// binary searched.
    pub valid: Range<u32>,
    /// The CST node for the component holding this child, or `INVALID_ID` once it has been removed
    /// as a centroid.
    pub node: u32,
}

/// The Centroid Search Trees of all tokens, flattened into shared arenas.
pub(crate) struct CentroidSearchTree {
    pub nodes: Vec<CentroidSearchTreeNode>,
    pub children: Vec<CentroidSearchTreeChild>,
    /// `roots[dfs_in(τ)]` is where `descend` starts. It is keyed by timestamp because `StateEntry`
    /// does not carry τ's id.
    pub roots: Vec<u32>,
}

/// Scratch space for building one SufSucTree(τ) at a time.
///
/// Every buffer is reused across tokens, and the ones indexed by local number are reinitialized
/// with `resize` for each of them.
struct Builder<'a> {
    table: &'a TokenTableBuilder,
    tree: CentroidSearchTree,
    /// Maps a local node number to a global token id.
    nodes: Vec<u32>,
    /// The parent in SufSucTree(τ), as a local number. `INVALID_ID` at the root.
    par: Vec<u32>,
    /// The child lists of SufSucTree(τ) in CSR form over local numbers.
    ch_start: Vec<u32>,
    ch_list: Vec<u32>,
    /// Cursor for the counting sort, which doubles as the CSR offsets.
    counts: Vec<u32>,
    /// Nodes already taken as centroids and thus cut off from the remaining components.
    removed: Vec<bool>,
    /// Subtree sizes within the current component, counted with the traversal's starting point as
    /// the root.
    size: Vec<u32>,
    /// For each node of the current component, the size of its largest child
    /// subtree.
    best_child: Vec<u32>,
    /// BFS order of the current component, which doubles as the list of its nodes.
    order: Vec<u32>,
    /// The parent within that BFS traversal. Distinct from `par`, because the component is
    /// traversed as an undirected graph and successor edges are sometimes followed upwards.
    trav_par: Vec<u32>,
    /// Maps a global token id to a local node number.
    ///
    /// Only the entries for the current S_τ are overwritten, and stale ones are never read. S_τ is
    /// closed under suc(·), so every lookup asks for a token this pass has just written.
    local_of: Vec<u32>,
}

impl CentroidSearchTree {
    /// Reorders the arenas so that the nodes with the largest `counts` come first.
    ///
    /// For the layout optimization of `IncrementalBpeBuilder::corpus`. Nothing observable changes,
    /// since everything that refers to a position (`roots`, `up_component`, `node` and
    /// `child_start`) is rewritten here. What changes is locality. Visits concentrate on a small
    /// hot set, so the dependent loads of `descend` come to fit in cache. The sort is stable, so
    /// the result stays deterministic.
    ///
    /// # Arguments
    ///
    /// * `counts` - Visit count per node. Same length as `nodes`.
    pub(crate) fn reorder_by_counts(&mut self, counts: &[u64]) {
        let n = u32::try_from(self.nodes.len()).unwrap();
        let mut order: Vec<_> = (0..n).collect();
        order.sort_by_key(|&i| core::cmp::Reverse(counts[usize::from_u32(i)]));
        // new_of_old[old] = the new position of the node that was at `old`.
        let mut new_of_old = vec![0u32; self.nodes.len()];
        for (new, &old) in order.iter().enumerate() {
            new_of_old[usize::from_u32(old)] = u32::try_from(new).unwrap();
        }
        let mut nodes = Vec::with_capacity(self.nodes.len());
        let mut children = Vec::with_capacity(self.children.len());
        for &old in &order {
            let mut node = self.nodes[usize::from_u32(old)];
            let start = usize::from_u32(node.child_start);
            let end = start + usize::from_u32(node.child_len);
            node.child_start = u32::try_from(children.len()).unwrap();
            for &child in &self.children[start..end] {
                children.push(CentroidSearchTreeChild {
                    node: if child.node == INVALID_ID {
                        INVALID_ID
                    } else {
                        new_of_old[usize::from_u32(child.node)]
                    },
                    ..child
                });
            }
            if node.up_component != INVALID_ID {
                node.up_component = new_of_old[usize::from_u32(node.up_component)];
            }
            nodes.push(node);
        }
        // The child slices are disjoint and cover the whole arena, so nothing is dropped or
        // duplicated.
        for root in &mut self.roots {
            *root = new_of_old[usize::from_u32(*root)];
        }
        self.nodes = nodes;
        self.children = children;
    }

    /// Precomputes one Centroid Search Tree per canonical token.
    ///
    /// # Arguments
    ///
    /// * `table` - The linearized vocabulary.
    /// * `pma` - The automaton, reused to enumerate the suffix tokens.
    pub(crate) fn build(
        table: &TokenTableBuilder,
        pma: &DoubleArrayAhoCorasick<StateEntry>,
    ) -> Self {
        let k = table.len();
        let k32 = u32::try_from(k).unwrap();
        let mut builder = Builder {
            table,
            tree: CentroidSearchTree {
                nodes: vec![],
                children: vec![],
                roots: vec![INVALID_ID; k],
            },
            nodes: vec![],
            par: vec![],
            ch_start: vec![],
            ch_list: vec![],
            counts: vec![],
            removed: vec![],
            size: vec![],
            best_child: vec![],
            order: vec![],
            trav_par: vec![],
            local_of: vec![0; k],
        };

        // The payload carries no token id, so invert dfs_in to recover it.
        let mut token_of_dfs = vec![0u32; k];
        for t in 0..k32 {
            token_of_dfs[usize::from_u32(table.dfs_in[usize::from_u32(t)])] = t;
        }
        let mut suffixes = vec![];
        for tau in 0..k32 {
            // Feeding τ through the automaton and reading the overlapping matches at the final
            // position enumerates exactly the vocabulary entries that are suffixes of τ.
            let mut stepper = pma.find_overlapping_stepper();
            for &byte in table.token_bytes(tau) {
                stepper.consume(byte);
            }
            suffixes.clear();
            suffixes.extend(
                stepper
                    .matches()
                    .map(|m| token_of_dfs[usize::from_u32(m.value().dfs_in_tau)]),
            );
            // Length order is a topological order of SufSucTree(τ), because suc(u) is a suffix of
            // τ strictly shorter than u and thus always comes before u. Local node 0 is the root.
            suffixes.sort_unstable_by_key(|&x| table.token_len[usize::from_u32(x)]);
            // τ is its own longest suffix, and the root is the atomic token of its last byte.
            builder.build_local(&suffixes);
            // Decomposition starts at the root; the first centroid found becomes the root of the
            // CST.
            let root = builder.decompose(0);
            builder.tree.roots[usize::from_u32(table.dfs_in[usize::from_u32(tau)])] = root;
        }

        builder.tree
    }
}

impl Builder<'_> {
    /// Builds SufSucTree(τ) over local numbers and initializes the scratch space for the
    /// decomposition.
    ///
    /// # Arguments
    ///
    /// * `suffixes` - The token ids of S_τ in topological (length) order.
    fn build_local(&mut self, suffixes: &[u32]) {
        let m = suffixes.len();
        self.nodes.clear();
        self.nodes.extend_from_slice(suffixes);
        for (local, &tok) in suffixes.iter().enumerate() {
            self.local_of[usize::from_u32(tok)] = u32::try_from(local).unwrap();
        }

        self.par.clear();
        self.par.resize(m, INVALID_ID);
        self.counts.clear();
        self.counts.resize(m + 1, 0);
        for local in 0..m {
            let tok = self.nodes[local];
            if self.table.is_atomic(tok) {
                continue;
            }
            let suc = self.table.suc[usize::from_u32(tok)];
            let p = self.local_of[usize::from_u32(suc)];
            self.par[local] = p;
            self.counts[usize::from_u32(p) + 1] += 1;
        }
        for i in 0..m {
            self.counts[i + 1] += self.counts[i];
        }
        self.ch_start.clear();
        self.ch_start.extend_from_slice(&self.counts);
        self.ch_list.clear();
        self.ch_list.resize(m.saturating_sub(1), 0);
        for local in 1..m {
            let p = usize::from_u32(self.par[local]);
            let pos = self.counts[p];
            self.ch_list[usize::from_u32(pos)] = u32::try_from(local).unwrap();
            self.counts[p] = pos + 1;
        }
        for u in 0..m {
            let start = usize::from_u32(self.ch_start[u]);
            let end = usize::from_u32(self.ch_start[u + 1]);
            self.ch_list[start..end].sort_unstable_by_key(|&v| {
                self.table.search[usize::from_u32(self.nodes[usize::from_u32(v)])]
                    .valid
                    .start
            });
        }

        self.removed.clear();
        self.removed.resize(m, false);
        self.size.clear();
        self.size.resize(m, 0);
        self.best_child.clear();
        self.best_child.resize(m, 0);
        self.trav_par.clear();
        self.trav_par.resize(m, INVALID_ID);
    }

    /// Decomposes a component recursively and returns the index of the
    /// corresponding CST node.
    ///
    /// Removing the centroid splits the component into an upward one and one per child, each at
    /// most half the original size, so the recursion depth is O(log |τ|).
    ///
    /// # Arguments
    ///
    /// * `entry` - Local number of a node in the component to decompose.
    fn decompose(&mut self, entry: u32) -> u32 {
        let centroid = usize::from_u32(self.find_centroid(entry));
        self.removed[centroid] = true;

        let idx = u32::try_from(self.tree.nodes.len()).unwrap();
        let token = self.nodes[centroid];
        self.tree.nodes.push(CentroidSearchTreeNode {
            token,
            up_component: INVALID_ID,
            child_start: u32::try_from(self.tree.children.len()).unwrap(),
            child_len: (self.ch_start[centroid + 1] - self.ch_start[centroid]),
            search: self.table.search[usize::from_u32(token)],
        });

        let start = usize::from_u32(self.ch_start[centroid]);
        let end = usize::from_u32(self.ch_start[centroid + 1]);
        let base = self.tree.children.len();
        for j in start..end {
            let v = self.ch_list[j];
            let child_token = usize::from_u32(self.nodes[usize::from_u32(v)]);
            self.tree.children.push(CentroidSearchTreeChild {
                valid: self.table.search[child_token].valid,
                node: INVALID_ID,
            });
        }
        for (offset, j) in (start..end).enumerate() {
            let v = self.ch_list[j];
            if !self.removed[usize::from_u32(v)] {
                let child_node = self.decompose(v);
                self.tree.children[base + offset].node = child_node;
            }
        }
        let p = self.par[centroid];
        if p != INVALID_ID && !self.removed[usize::from_u32(p)] {
            let up_node = self.decompose(p);
            self.tree.nodes[usize::from_u32(idx)].up_component = up_node;
        }
        idx
    }

    /// Returns the centroid of a component, i.e. the node whose removal minimizes the largest
    /// remaining piece.
    ///
    /// Written iteratively rather than recursively, scanning the component linearly three times,
    /// which makes the whole decomposition O(|S_τ| log |S_τ|).
    ///
    /// # Arguments
    ///
    /// * `entry` - Local number of a node in the component.
    fn find_centroid(&mut self, entry: u32) -> u32 {
        self.order.clear();
        self.order.push(entry);
        self.trav_par[usize::from_u32(entry)] = INVALID_ID;
        let mut i = 0;
        while i < self.order.len() {
            let v = usize::from_u32(self.order[i]);
            i += 1;
            let came_from = self.trav_par[v];
            let p = self.par[v];
            if p != INVALID_ID && p != came_from && !self.removed[usize::from_u32(p)] {
                self.trav_par[usize::from_u32(p)] = u32::try_from(v).unwrap();
                self.order.push(p);
            }
            let start = usize::from_u32(self.ch_start[v]);
            let end = usize::from_u32(self.ch_start[v + 1]);
            for j in start..end {
                let w = self.ch_list[j];
                if w != came_from && !self.removed[usize::from_u32(w)] {
                    self.trav_par[usize::from_u32(w)] = u32::try_from(v).unwrap();
                    self.order.push(w);
                }
            }
        }

        let total = u32::try_from(self.order.len()).unwrap();
        for &v in &self.order {
            self.size[usize::from_u32(v)] = 1;
            self.best_child[usize::from_u32(v)] = 0;
        }
        for idx in (1..self.order.len()).rev() {
            let v = usize::from_u32(self.order[idx]);
            let p = usize::from_u32(self.trav_par[v]);
            self.size[p] += self.size[v];
        }
        for idx in (1..self.order.len()).rev() {
            let v = usize::from_u32(self.order[idx]);
            let p = usize::from_u32(self.trav_par[v]);
            self.best_child[p] = self.best_child[p].max(self.size[v]);
        }

        let mut best = entry;
        let mut best_piece = u32::MAX;
        for &v in &self.order {
            let vi = usize::from_u32(v);
            let piece = self.best_child[vi].max(total - self.size[vi]);
            if piece < best_piece {
                best_piece = piece;
                best = v;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use daachorse::DoubleArrayAhoCorasickBuilder;

    use crate::vocab::TokenTableBuilder;
    use crate::vocab::tests::sample_vocabulary;

    use super::{CentroidSearchTree, INVALID_ID, StateEntry};

    /// Builds the automaton and the Centroid Search Trees for `vocab`, the way
    /// `IncrementalBpeBuilder::build` does.
    ///
    /// # Arguments
    ///
    /// * `vocab` - A proper token list.
    fn build_trees(vocab: &[Vec<u8>]) -> (TokenTableBuilder, CentroidSearchTree) {
        let table = TokenTableBuilder::new(vocab).unwrap();
        let k = u32::try_from(table.len()).unwrap();
        let patterns = (0..k).map(|i| (table.token_bytes(i).to_vec(), table.to_entry(i)));
        let pma = DoubleArrayAhoCorasickBuilder::new()
            .build_with_values::<_, _, StateEntry>(patterns)
            .unwrap();
        let tree = CentroidSearchTree::build(&table, &pma);
        (table, tree)
    }

    /// The CST nodes reachable from `root`, following both the upward component and the child
    /// components.
    ///
    /// # Arguments
    ///
    /// * `tree` - The arena to walk.
    /// * `root` - Index of the node to start from.
    fn reachable(tree: &CentroidSearchTree, root: u32) -> Vec<u32> {
        let mut seen = vec![root];
        let mut i = 0;
        while i < seen.len() {
            let node = tree.nodes[usize::try_from(seen[i]).unwrap()];
            i += 1;
            let start = usize::try_from(node.child_start).unwrap();
            let end = start + usize::try_from(node.child_len).unwrap();
            let mut next: Vec<u32> = tree.children[start..end]
                .iter()
                .map(|child| child.node)
                .collect();
            next.push(node.up_component);
            for candidate in next {
                if candidate != INVALID_ID && !seen.contains(&candidate) {
                    seen.push(candidate);
                }
            }
        }
        seen
    }

    /// Asserts that the children of every centroid in `tree` are sorted by the start of their valid
    /// interval and pairwise disjoint, returning how many sibling pairs it compared.
    ///
    /// # Arguments
    ///
    /// * `tree` - The arena to check.
    fn count_checked_sibling_pairs(tree: &CentroidSearchTree) -> usize {
        let mut checked = 0;
        for node in &tree.nodes {
            let start = usize::try_from(node.child_start).unwrap();
            let end = start + usize::try_from(node.child_len).unwrap();
            for pair in tree.children[start..end].windows(2) {
                assert!(pair[0].valid.end <= pair[1].valid.start,);
                checked += 1;
            }
        }
        checked
    }

    /// Runs [`count_checked_sibling_pairs`] over `seeds` vocabularies.
    ///
    /// # Arguments
    ///
    /// * `seeds` - How many vocabularies to sweep.
    fn check_sibling_order(seeds: u64) -> usize {
        let mut checked = 0;
        for seed in 1..=seeds {
            let (_, tree) = build_trees(&sample_vocabulary(seed));
            checked += count_checked_sibling_pairs(&tree);
        }
        checked
    }

    /// Asserts that the CST of `tau` holds each of its suffix tokens exactly once, returning 1 when
    /// the tree was large enough to be interesting.
    ///
    /// # Arguments
    ///
    /// * `table` - The linearized vocabulary.
    /// * `tree` - The arena holding every CST.
    /// * `tau` - Internal id of the token whose CST is checked.
    fn check_one_tree(table: &TokenTableBuilder, tree: &CentroidSearchTree, tau: u32) -> usize {
        let bytes = table.token_bytes(tau);
        let k = u32::try_from(table.len()).unwrap();
        // S_τ by brute force, i.e. the canonical tokens that are suffixes of τ.
        let want = (0..k)
            .filter(|&other| bytes.ends_with(table.token_bytes(other)))
            .count();

        let root =
            tree.roots[usize::try_from(table.dfs_in[usize::try_from(tau).unwrap()]).unwrap()];
        let mut tokens: Vec<u32> = reachable(tree, root)
            .iter()
            .map(|&i| tree.nodes[usize::try_from(i).unwrap()].token)
            .collect();
        tokens.sort_unstable();
        let reached = tokens.len();
        tokens.dedup();
        assert_eq!(tokens.len(), reached,);
        assert_eq!(tokens.len(), want,);
        usize::from(want > 2)
    }

    /// Runs [`check_one_tree`] for every token of `seeds` vocabularies, and
    /// returns how many non-trivial trees it saw.
    ///
    /// # Arguments
    ///
    /// * `seeds` - How many vocabularies to sweep.
    fn check_tree_membership(seeds: u64) -> usize {
        let mut checked = 0;
        for seed in 1..=seeds {
            let (table, tree) = build_trees(&sample_vocabulary(seed));
            for tau in 0..u32::try_from(table.len()).unwrap() {
                checked += check_one_tree(&table, &tree, tau);
            }
        }
        checked
    }

    /// `descend` binary-searches the children of a centroid, so they must be
    /// sorted by the start of their valid interval. Sibling intervals are mutually exclusive, so
    /// they must also be disjoint.
    #[test]
    fn centroid_children_are_sorted_and_disjoint() {
        let checked = check_sibling_order(4);
        assert!(checked > 0,);
    }

    /// Decomposing SufSucTree(τ) must place every one of its nodes in exactly
    /// one component, so each suffix token appears exactly once in the tree
    /// reachable from that root.
    #[test]
    fn each_centroid_tree_holds_every_suffix_token_once() {
        let checked = check_tree_membership(4);
        assert!(checked > 0);
    }
}
