"""Toy implementation of the GGH signature scheme using sympy

Signing is done by solving the CVP problem using Babai's nearest plane so that
the verification radius can be efficiently computed
"""
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form
import numpy as np

from .arithmetic import gram_schmidt, nearest_plane, ortho_defect

UNDERFLOW_THRESHOLD = 1e-10


class LatticeBasis:
    """Full rank basis of lattice in R^n"""

    def __init__(self, vectors: np.ndarray):
        """Non-singularity is checked by np.linalg.inv so there is no need to
        explicitly check
        """
        self.vectors = vectors
        self.vectors_orth = gram_schmidt(vectors)
        self.vectors_inv = np.linalg.inv(vectors)
        self.ortho_defect = ortho_defect(vectors)

    def iter_base_vectors(self, take: int | None = None):
        """Iterate through the column vectors of self.vectors

        If take is supplied, then only yield up to the first (take) vectors
        """
        cnt = 0
        for i in range(self.vectors.shape[1]):
            if (take is not None) and (cnt >= take):
                return
            yield self.vectors[:, i]
            cnt += 1

    def to_sympy_matrix(self):
        """Convenient function that returns a copy of self.vectors as sympy
        matrix
        """
        if self.is_zz():
            # TODO: higher dimensions will cause integer overflow when numpy
            #   tries to convert Python's big integer into C int
            # for now, n = 6 is the highest we can go before HNF overflows
            # C integers
            return Matrix(self.vectors.astype(int))
        return Matrix(self.vectors)

    @property
    def dim(self):
        return self.vectors.shape[0]

    def cvp_with_rounding(self, target: np.ndarray):
        """Solve the CVP problem using simple rounding"""
        coordinates = np.round(self.vectors_inv.dot(target))
        return self.vectors.dot(coordinates)

    def cvp_with_nearest_plane(self, target: np.ndarray):
        """Solve the CVP problem using Nearest Plane"""
        return nearest_plane(target, self.vectors, self.vectors_orth)

    def is_zz(self) -> bool:
        """Return True iff all entries are no further from its nearest integer
        than the specified threshold
        """
        for r in self.vectors:
            for val in r:
                if np.round(val) - val > UNDERFLOW_THRESHOLD:
                    return False
        return True

    def dist(self, other: "LatticeBasis"):
        """A way to measure the distance between two lattices, computed by the
        determinant of the matrix (A_inv)B, where A, B are the two basis.
        The closer this value is to 1, the closer the two lattices are. If the
        value is 1, then the two lattices are identical, up to some rotation
        """
        return np.linalg.det(self.vectors_inv.dot(other.vectors))

    def is_equiv(self, other: "LatticeBasis") -> bool:
        """True if and only if the two basis are related by a unimodular
        matrix
        """
        relation = self.to_sympy_matrix().inv() * other.to_sympy_matrix()
        for val in relation:
            if not val.is_integer:
                return False
        if relation.det() != 1:
            return False
        return True


class PublicKey(LatticeBasis):
    """Highly skewed basis of the same lattice as the secret key"""

    pass


class SecretKey(LatticeBasis):
    """Highly orthogonal basis"""

    @staticmethod
    def generate_sk(n: int, magnitude: int = 100):
        """Generate a secret key given the input number of dimensions
        sk = R = kI + R'
        where R' is sampled from [-l, ..., l-1] and
        k = sqrt(n) * l
        """
        l = magnitude
        k = round(np.sqrt(n) + 1) * l + 1
        vectors = np.eye(n) * k + np.random.randint(-l, l, size=(n, n))
        return SecretKey(vectors)

    def generate_public_key(self):
        """The public key is the Hermite normal form. This requires the entries
        to be integral
        """
        if not self.is_zz():
            raise ValueError("Secret basis is not integral")
        hnf = hermite_normal_form(Matrix(self.vectors.astype(int)))
        vectors = np.array(hnf).astype(float)
        return PublicKey(vectors)


def generate_keypair(n: int):
    """Generate a pair of secret key and public key"""
    sk = SecretKey.generate_sk(n)
    pk = sk.generate_public_key()

    if not pk.is_equiv(sk):
        raise ValueError("Keypair is inconsistent")

    return pk, sk


def keygen(n: int):
    """Generate a key pair from the security parameters."""
    l = 100
    k = round(np.sqrt(n) + 1) * l + 1

    sk = (np.eye(n) * k + np.random.randint(-l, l, size=(n, n))).astype(int)
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
    """Output the lattice point closest to message using nearest plane"""
    coord = nearest_plane(message, sk)
    # coord = cvp_round(message, sk)
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
    return dist < radius
