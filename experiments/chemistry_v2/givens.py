"""Givens-tree angle encoding used by the verifier (decode in torch, encode in numpy)."""

from __future__ import annotations

import numpy as np
import torch


def decode_torch(theta_over_pi: torch.Tensor, edges: np.ndarray) -> torch.Tensor:
    """Apply the ordered tree rotations to e_0: (M, N-1) -> (M, N) unit rows."""
    m, num_angles = theta_over_pi.shape
    cols = [None] * (num_angles + 1)
    cols[0] = torch.ones(m, dtype=theta_over_pi.dtype, device=theta_over_pi.device)
    zero = torch.zeros_like(cols[0])
    theta = torch.pi * theta_over_pi
    cos, sin = torch.cos(theta), torch.sin(theta)
    for t, (p, c) in enumerate(edges):
        vp = cols[p]
        vc = cols[c] if cols[c] is not None else zero
        cols[p] = cos[:, t] * vp - sin[:, t] * vc
        cols[c] = sin[:, t] * vp + cos[:, t] * vc
    return torch.stack(cols, dim=1)


def decode_numpy(theta_over_pi: np.ndarray, edges: np.ndarray) -> np.ndarray:
    m, num_angles = theta_over_pi.shape
    v = np.zeros((m, num_angles + 1))
    v[:, 0] = 1.0
    for t, (p, c) in enumerate(edges):
        th = np.pi * theta_over_pi[:, t]
        vp, vc = v[:, p].copy(), v[:, c].copy()
        v[:, p] = np.cos(th) * vp - np.sin(th) * vc
        v[:, c] = np.sin(th) * vp + np.cos(th) * vc
    return v


def encode_numpy(u: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Exact inverse of ``decode_numpy`` up to the row sign: (M, N) -> (M, N-1) in (-1, 1]."""
    v = u / np.linalg.norm(u, axis=1, keepdims=True)
    v = v.copy()
    theta = np.zeros((v.shape[0], len(edges)))
    for t in range(len(edges) - 1, -1, -1):
        p, c = edges[t]
        th = np.arctan2(v[:, c], v[:, p])
        r = np.hypot(v[:, p], v[:, c])
        theta[:, t] = th / np.pi
        v[:, p] = r
        v[:, c] = 0.0
    # Root sign: v[:, 0] is now +1 (r >= 0), so decode reproduces u exactly.
    return theta


def to_words(theta_over_pi: np.ndarray, brot: int) -> np.ndarray:
    scale = 1 << (brot - 1)
    return np.mod(np.rint(theta_over_pi * scale).astype(np.int64), 1 << brot)


def words_to_theta(words: np.ndarray, brot: int) -> np.ndarray:
    return words.astype(np.float64) / (1 << (brot - 1))
