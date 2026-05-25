"""
Permute sentence-level trajectory order before SDS fitting.

What gets shuffled
------------------
Each trajectory is a list of rows (T timesteps), one row per CoT sentence from
``create_dataset.py``:

  timestep t  ->  (CEBRA embedding z_t, PCA row p_t, stage label L_t)

All three are permuted with the **same** index permutation ``perm``. Row t always
belongs to the **same sentence** as in the original trace; only its **position**
in the sequence fed to SDS changes.

Example (one problem, full shuffle)::

  Real order (sentence text -> row index):
    0: "Let me set up..."     -> z_0
    1: "We compute..."        -> z_1
    2: "Therefore answer=12" -> z_2

  perm = [2, 0, 1]
  Shuffled sequence passed to SDS: [z_2, z_0, z_1]

What this tests
---------------
Whether SDS metrics (persistence, transition structure, predictive R^2) depend on
**genuine temporal order** of reasoning steps. If metrics collapse under shuffle,
the readout is using sequential structure rather than order-invariant clusters alone.
"""

from __future__ import annotations

import numpy as np


def permute_trajectory(seq: np.ndarray, rng: np.random.Generator, mode: str = "full", block_size: int = 3) -> np.ndarray:
    """Return a row-permuted copy of ``seq`` (shape [T, D]).

    Row ``i`` is the activation (or embedding) for sentence ``i`` in the original
    CoT; after permutation it sits at a new timestep in the SDS input sequence.
    """
    n = len(seq)
    if n <= 1:
        return seq.copy()

    if mode == "full":
        perm = rng.permutation(n)
    elif mode == "block":
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        blocks = [np.arange(i, min(i + block_size, n)) for i in range(0, n, block_size)]
        block_order = rng.permutation(len(blocks))
        perm = np.concatenate([blocks[i] for i in block_order])
    else:
        raise ValueError(f"Unknown shuffle mode: {mode!r}")

    return seq[perm]


def shuffle_sequence_lists(
    cebra_seqs: list[np.ndarray],
    pca_seqs: list[np.ndarray],
    labels: list[list],
    seed: int,
    mode: str = "full",
    block_size: int = 3,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list]]:
    """Apply one permutation per trajectory, shared across CEBRA / PCA / labels.

    Parameters
    ----------
    cebra_seqs, pca_seqs
        One array per problem, shape ``[T, D]``, rows in real CoT sentence order.
    labels
        Per-timestep stage strings aligned with rows (e.g. PLAN_GENERATION).

    Returns
    -------
    Same structure with rows reordered. Example: if problem p has T=4 and
    ``perm = [3, 1, 0, 2]``, then ``cebra_out[p] = cebra_seqs[p][perm]`` and
    ``labels_out[p][t]`` is the stage that belonged to original sentence ``perm[t]``.
    """
    rng = np.random.default_rng(seed)
    cebra_out, pca_out, labels_out = [], [], []
    for z_seq, p_seq, lab in zip(cebra_seqs, pca_seqs, labels):
        n = len(z_seq)
        if n <= 1:
            cebra_out.append(z_seq.copy())
            pca_out.append(p_seq.copy())
            labels_out.append(list(lab))
            continue
        if mode == "full":
            perm = rng.permutation(n)
        elif mode == "block":
            blocks = [np.arange(i, min(i + block_size, n)) for i in range(0, n, block_size)]
            block_order = rng.permutation(len(blocks))
            perm = np.concatenate([blocks[i] for i in block_order])
        else:
            raise ValueError(f"Unknown shuffle mode: {mode!r}")
        cebra_out.append(z_seq[perm])
        pca_out.append(p_seq[perm])
        labels_out.append([lab[i] for i in perm])
    return cebra_out, pca_out, labels_out
