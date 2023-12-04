from sympy import randMatrix
from .signature import sign, verify

def generate_pairs(pk, sk, radius, ndim, npairs, val_limits=99999):
    pairs = []
    while len(pairs) < npairs:
        m = randMatrix(ndim, 1, min=-val_limits, max=val_limits)
        sigma = sign(m, sk)
        verified = verify(m, sigma, pk, radius)
        if verified:
            pairs.append((m, sigma))

    return pairs