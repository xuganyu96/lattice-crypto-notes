"""Toy implementation of the GGH signature scheme using sympy

Signing is done by solving the CVP problem using Babai's nearest plane so that
the verification radius can be efficiently computed
"""
import sympy
from sympy import Matrix, Float
from sympy.matrices.normalforms import hermite_normal_form
import numpy as np

from .arithmetic import gram_schmidt, nearest_plane

def keygen(n: int):
    """Generate a key pair from the security parameters. 
    """
    l = 10
    k = round(np.sqrt(n) * l)

    sk = (np.eye(n) * k + np.random.randint(-l, l+1, size=(n, n))).astype(int)
    pk = np.array(hermite_normal_form(Matrix(sk))).astype(int)

    u = Matrix(pk).inv() * Matrix(sk)
    if (not all([val.is_integer for val in u])) or (u.det() != 1):
        raise Exception("Key pair is inconsistent")
    return pk, sk, get_veri_radius(sk)

def get_veri_radius(sk: np.ndarray):
    """Given a secret key matrix, compute the distance within which the message
    and the signature can be considered valid. This radius is half of the
    n-dimensional diagonal, which we can easily compute using Pythagoras
    """
    sk_orth = gram_schmidt(sk)
    diag = 0
    for col in range(sk_orth.shape[1]):
        col = sk_orth[:, col]
        diag += col.dot(col)
    radius = np.sqrt(diag) / 2
    return radius

def sign(
    message: np.ndarray,
    sk: np.ndarray,
) -> np.ndarray:
    """Output the lattice point closest to message using nearest plane
    """
    coord = nearest_plane(message, sk)
    return sk.dot(coord)

def verify(
    message: np.ndarray,
    sigma: np.ndarray,
    pk: np.ndarray,
    radius: float,
) -> bool:
    """Verify the signature by checking that it is indeed a lattice point and
    that the message is within the verification radius
    """
    coord = Matrix(pk).inv() * Matrix(sigma)
    if not all([val.is_integer for val in coord]):
        return False
    dist = np.linalg.norm(message - sigma)
    print(message, sigma)
    print(dist, radius)
    return dist < radius