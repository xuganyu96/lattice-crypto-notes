from __future__ import annotations
import numpy as np
from sympy import Matrix


def ortho_defect(basis: np.ndarray):
    prod = 1
    for col in range(basis.shape[1]):
        col = basis[:, col]
        prod *= np.linalg.norm(col)
    defect = prod / np.linalg.det(basis)
    return defect


def gram_schmidt(basis: np.ndarray) -> np.ndarray:
    ortho_basis = []
    for col in range(basis.shape[1]):
        col = basis[:, col]

        proj_sum = np.zeros_like(col).astype(float)
        for ortho_base in ortho_basis:
            numerator = col.dot(ortho_base)
            denominator = ortho_base.dot(ortho_base)
            mu = numerator / denominator
            proj = mu * ortho_base
            proj_sum += proj

        ortho_basis.append(col.astype(float) - proj_sum)

    return np.array(ortho_basis).transpose()


def nearest_plane(
    target: np.ndarray, basis: np.ndarray, ortho_basis: np.ndarray | None = None
) -> np.ndarray:
    """Return the coordinate of the lattice point v such that "target - v" is
    in "v + orthogonal fundamental parallelepiped"
    """
    if basis.shape[1] == 0:
        return np.array([])
    if ortho_basis is None:
        ortho_basis = gram_schmidt(basis)
    last_col = basis[:, -1]
    last_ortho_col = ortho_basis[:, -1]
    c = round((target.dot(last_ortho_col) / last_ortho_col.dot(last_ortho_col)))

    return np.concatenate(
        [
            nearest_plane(target - c * last_col, basis[:, :-1], ortho_basis[:, :-1]),
            np.array([c]),
        ]
    ).astype(int)


def cvp_round(
    target: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Return the coordinate of the lattice point v such that (target - v) is
    in the centered fundamental parallelpiped
    """
    return np.around(np.linalg.inv(basis).dot(target))
