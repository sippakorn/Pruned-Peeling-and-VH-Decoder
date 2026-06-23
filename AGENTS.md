# AGENTS.md — PPVH-Connolly

> Repo context + working notes for any agent picking up this project. Covers the codebase
> map, the documentation deliverable, the concepts that were clarified, verified facts, and
> gotchas. Written 2026-06-22.

## What this project is

Python implementation of the **Pruned-Peeling and VH (vertical/horizontal) erasure decoder**
for **hypergraph-product (HGP) quantum LDPC codes**, from the paper:
Connolly, Londe, Leverrier & Delfosse, *"Fast erasure decoder for a class of quantum LDPC
codes"*, [arXiv:2208.01002](http://arxiv.org/abs/2208.01002) (2022).

Pipeline: build an HGP code from two classical codes → erase a known set of qubits → recover
the error from the syndrome via a cascade of tricks (plain peeling → VH pruned peeling →
cluster decoder) → Monte-Carlo the failure rate vs erasure rate.

## File map

| File | Role | Key symbols |
| --- | --- | --- |
| `Hypergraph_Product_Code_Construction_v3.py` | Builds the quantum code | `HGP_code`, `Check`, `Generator`, `standard_form` (GE), `index_to_biindex`, `Toric3` example, syndrome + logical tests |
| `utilities.py` | Helpers | random erasures, adjacency lists, components/clusters, `perform_classical_syndrome_analysis_with_erasure`, `construct_HGP_code_from_classical_H_text_file` |
| `peeling_cluster_decoder.py` | The decoders | `combined_peeling_and_cluster_decoder`, `cluster_decoder`, `solve_cluster_tree_by_recursive_peeling`, simulation harness, `main()` |
| `PEG_HGP_code_(3,4)_family_*.txt` | Sample classical H matrices (adjacency-list format) seeding real codes (625–2025 qubits) | first line = `num_checks num_bits` |

## Deliverable produced: `PROJECT_EXPLAINED.html`

Self-contained, **light-themed**, single-page explainer (no external deps — MathJax deliberately
avoided so it works offline). Per-concept pattern: **Key idea → pen-and-paper example → code pointer**.
14 sections (GF2 → Tanner graphs → erasure → CSS → HGP construction → index/biindex → logical
errors → peeling → VH → clusters → recursive peeling → Gaussian elimination → simulation → how to run).

Extra callouts added during refinement (these answer the user's specific questions):

- **§1** "XOR view of H·e on the flat 1-D array" + a code box showing dense (`Hz.dot(e)%2`) vs sparse
  (`len(intersection)%2`) syndrome.
- **§5** "Reading the stride in Hz" — decomposes the Kronecker `⊗I` stride structure.
- **§6** "Why two coordinate systems?" and "Why the horizontal/vertical test matters".
- **§7** rewritten to build **classical → quantum**: logical operators & logical errors.

### HTML conventions (follow these when editing)

- Embedded CSS, light theme. Box classes: `.idea` (blue, "Key idea"), `.pnp` (amber, "Pen & paper"),
  `.codept` (green, "In the code"), `.note` (pink, "heads-up"). `.matrix` = monospace block,
  `.fileref` = code-location label.
- Escape `<` `>` `&` inside `<pre><code>` blocks.
- **Validate after every edit:**
  `python3 -c "import html.parser as h; h.HTMLParser().feed(open('PROJECT_EXPLAINED.html').read()); print('OK')"`

## Concepts clarified (the substance of the discussion)

1. **GF(2) / parity**: `H·e mod 2` is row-wise XOR; syndrome `s = H·e`. XOR cancellation (1⊕1=0) is why
   decoding is hard.
2. **HGP construction**: `Hz = [ I⊗H2 | H1ᵀ⊗I ]`, `Hx = [ H1⊗I | I⊗H2ᵀ ]`. Qubits split into
   **horizontal** (B1×B2, `n1·n2`, indexed first) and **vertical** (C1×C2, `r1·r2`, indexed after).
3. **CSS commutation**: `Hx·Hzᵀ = 0` (checks share an even # of qubits). Decodes X-errors with `Hz` only.
4. **Logical operators & errors** (§7, the classical→quantum bridge):
   - *Classical*: logical operator = nonzero codeword ∈ `ker(H)`; logical error = residual `e⊕ê` is a
     nonzero codeword (decoded to wrong codeword). Residual always has zero syndrome.
   - *Quantum twist*: stabilizers (rows of `Hx`) are harmless zero-syndrome residuals. So the failure
     test gains a 2nd condition: nontrivial logical iff `Hz·v = 0` **and** `Gx·v ≠ 0`.
5. **Decoder cascade**: M=0 dangling checks → M=1 fully-erased generators → M=2 erased generator
   products (VH trick to break cycles for free) → cluster decoder (recursive peeling of a cluster tree;
   declares failure on a cluster *cycle*).

### Deeper insights the user worked through (preserve these — they were the hard part)

- **1-D vs 2-D coordinates** (§6): 1-D is *forced* by linear algebra (each qubit = one column of `Hz`;
  `Hz·e` needs a single index) and because the two differently-shaped blocks need one uniform container
  (hence `index_shift`). 2-D `(row,col)` biindex is needed for the decoder's *geometry*. Convert on
  demand via cheap floor/mod: `index = row·num_cols + col`. Same erasure `{0,1,2}` is meaningless in 1-D
  but is obviously "the top row" in 2-D.

- **horizontal vs vertical test** (`qubit_index < num_h_qubits`): needed in 3 places — (a) different grid
  shapes ⇒ different biindex arithmetic (`num_cols=n2,shift=0` vs `num_cols=r2,shift=num_h_qubits`);
  (b) clusters group along rows (horizontal) vs columns (vertical); (c) a cluster maps to a *different
  classical code* — horizontal → solved in `H2` along its row, vertical → solved in `H1ᵀ` along its
  column. See `cluster_syndrome_analysis_with_zeroing` (peeling_cluster_decoder.py:193, 260–290).

- **Stride pattern** (§5): each `⊗I` lays a pattern at stride = grid row-length `num_cols` (row-major
  flattening). `I⊗H2` = contiguous copies offset by `n2`; `H1ᵀ⊗I` = one pattern spread at stride `r2`.
  Worked example: check 0 touches `{0,1,9,15}` = `{0,1}` (contiguous, H2 row 0) + `{9,15}` (strided,
  H1ᵀ row 0; `15−9 = 2·r2`).

- **Stride pattern vs sparse GE** — *different categories*: stride = a structural property (where the 1s
  are, forward map `H·e`); GE = an algorithm (backward map, solve `H·e=s`). They overlap **only** at the
  cluster solve (structure decides *which small matrix* GE runs on). Tension: GE causes **fill-in** and
  destroys the sparse stride structure — which is exactly *why* the decoder avoids global GE and only
  runs GE on tiny stuck clusters.

- **CONFIRMED: no sparse-matrix GE solver exists in this project.** Verified by source inspection:
  - Imports are only `numpy, random, itertools, copy, math, ast, sys, os, datetime`. No `scipy`,
    `scipy.sparse`, `galois`, `ldpc`, `networkx`, `sympy`, or `np.linalg`. (All "sparse" hits are
    comments / matplotlib strings.)
  - The only GE routine is `standard_form` (Hypergraph_..._v3.py:71) — a **dense** GF(2) Gauss-Jordan
    (`np.outer` + full-submatrix XOR `M[:, j:] = M[:, j:] ^ flip`). No CSR/CSC/COO/DOK/LIL, no `nnz`/
    `indptr`, no fill-in-aware pivoting, no `inv`/`lstsq`/`solve`/`spsolve`.
  - Called only at code setup (build `Gx`/`Gz`, lines 290–293) and inside the small per-cluster solve
    (`perform_classical_syndrome_analysis_with_erasure`, utilities.py:975).
  - The decoder's speed comes from **graph/adjacency-set** ops (peeling), not from GE.

## Verified concrete facts (produced by running the code)

- `Hrep3 = [[1,1,0],[0,1,1],[1,0,1]]`; `e=(1,0,1)` → syndrome `(1,1,0)`.
- `Toric3 = HGP_code(Hrep3, Hrep3)`: **18 qubits**, `kx=kz=10`, `dim=2` → a `[[18,2,3]]` code.
  `num_h_qubits = num_v_qubits = num_checks = num_generators = 9`. `Hx·Hzᵀ = 0` ✓.
- Check adjacencies (Toric3): `check0→{0,1,9,15}`, `check1→{1,2,10,16}`, `check2→{0,2,11,17}`, …
- Peeling-only run: `E={0,1}, e={0}, s={0,2}` → predicts `{0}`, `total_dangling_checks_corrected=2`.
- Classical-stopping-set run: `E={0,1,2}, e={1}, s={0,1}` → predicts `{1}`, `classical_stopping_sets_used=1`.
- XOR demo: `e={0,1}` → syndrome `{1,2}` (check 0 cancels: two flipped bits → XOR 0).
- Logical-error demo: stabilizer residual `{0,3,9,11}` (an `Hx` row) is harmless (both tests 0); the
  weight-3 set `{0,1,2}` is a true logical (`Hz·v=0`, `Gx·v={0,1,2}≠0`). `ker(Hrep3)={000,111}`; `111`
  is the classical logical operator; errors `{0}` and `{1,2}` share syndrome `(1,0,1)` and differ by `111`.

## Gotchas / how to run

**`peeling_cluster_decoder.py` ends with a bare `main()` (line 1222), no `if __name__` guard.**
Merely *importing* it launches the full 25 000-trial × 16-rate × 4-code simulation (runs a very
long time). For interactive work, copy the 3 modules to a temp dir and strip the trailing `main()`:

```bash
cp Hypergraph_Product_Code_Construction_v3.py utilities.py peeling_cluster_decoder.py /tmp/ppvh/
# then remove the final "main()" line from the temp copy before importing
```

(This is how the verified examples above were generated.) Quick toric example after neutralising `main()`:

```python
from Hypergraph_Product_Code_Construction_v3 import Toric3
from peeling_cluster_decoder import combined_peeling_and_cluster_decoder
s = Toric3.Hz_syn_index_set_for_X_err({0})
pred, results = combined_peeling_and_cluster_decoder(Toric3, {0,1}, s)   # -> {0}, dangling=2
```

## Peeling -> Sparse GE experiment (added 2026-06-23)

Tests the proposal: does DFS reordering speed up the sparse-GE residual solve? Two campaigns exist:

- **OLD / cascade** (`experiment_sparse_high_erasure_main`, tag `HIGHERASURE_`, report
  `sparse_high_erasure_report.html`): runs the FULL cascade (`combined_peeling_and_cluster_decoder`:
  M0 peeling -> M1 -> M2 -> cluster decoder) and only swaps the GE backend inside the *per-cluster*
  solves. This does NOT match the "jump straight from peeling to GE" proposal.
- **NEW / peeling->GE** (`experiment_peeling_ge_high_erasure_main`, CLI `peelge`, tag `PEELGEHIGH_`,
  report `peeling_ge_high_erasure_report.html`): the correct minimal pipeline. `peeling_then_ge_decoder`
  does M=0 dangling-check peeling ONLY, then feeds the whole residual straight into ONE
  `perform_classical_syndrome_analysis_with_erasure(HGP_code.Hz, s_vec, E)`. Backend/reorder come from
  `utilities.SOLVER_CONFIG`; the only step between peeling and GE is the optional DFS reorder. Conditions:
  `sparse` (natural order) vs `sparse_dfs` (DFS reorder). High regime 0.30->0.45, 200 trials, seed 12345.
  Run: `python peeling_cluster_decoder.py peelge` then `python generate_peeling_ge_report.py`.

**Result (this run):** summed over all rates+codes, DFS makes elimination ~1.04x faster but the reorder
cost makes the end-to-end pipeline ~1.59x SLOWER -> hypothesis NOT supported in this regime. (DFS also
slightly changes the failure rate because it selects a different valid coset representative of the solve.)

**Env gotcha:** workspace is a WSL share but the agent shell is Windows PowerShell. Calling `wsl.exe`
wedges the shell. Instead run with Windows Python over the UNC path (`numpy` was pip-installed for it).

## Open / offered but NOT done (potential next steps)

- Add a §12 HTML note explicitly stating "dense GF(2) Gauss-Jordan; no sparse solver" + the fill-in
  tension (user was offered this; not yet accepted).
- Add an `if __name__ == "__main__":` guard to `peeling_cluster_decoder.py` (offered; not done — repo
  not modified beyond adding `PROJECT_EXPLAINED.html` and this notes file).
