import time
import numpy as np
from .signature import SecretKey
    

def generate_pairs(sample_size: int, sk: SecretKey, maxval: int=100000):
    """Generate a large number of message, signature pairs
    """
    pairs = []
    start = time.monotonic()
    while len(pairs) < sample_size:
        m = np.random.randint(low=0, high=maxval, size=(sk.dim,))
        sigma = sk.cvp_with_rounding(m)
        pairs.append((m, sigma))
    dur = time.monotonic() - start
    print(f"Generated {sample_size} message-signature pairs in {dur:.2f} seconds")

    return pairs

