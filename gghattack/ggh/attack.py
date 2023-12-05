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

def approx_moment(w: np.ndarray, samples: list[np.ndarray], k: int):
    """Use a set of samples to approximate the k-th moment of the probability
    distribution over the input w
    """
    return (np.array(samples).dot(w) ** k).mean()

def approx_gradient(w: np.ndarray, samples: list[np.ndarray], k: int):
    """Approximate the gradient of the k-th moment at w using samples from the
    probability distribution
    """
    gradient = np.zeros_like(w).astype(float)
    dotproducts = k / len(samples) * (
        np.array(samples).dot(w) ** (k - 1)
    )
    for dotproduct, sample in zip(dotproducts, samples):
        gradient += dotproduct * sample
    return gradient


def gradient_descent(
    samples: list[np.ndarray],
    delta = 0.7,
    w_init: np.ndarray | None = None
):
    """Find the w that minimizes the fourth moment using gradient descent
    if an initial w is provided, then it will be used; otherwise a random w on
    the n-dimensional unit sphere will be sampled as an init point
    """
    k = 4
    w = np.random.randint(0, 10, size=(samples[0].shape[0],))
    w = w / np.linalg.norm(w)
    if w_init is not None:
        w = w_init
    # print(f"GradDesc began at {w}")
    iter = 0
    while True:
        # print(f"iteration {iter}, current loc is {w}")
        moment = approx_moment(w, samples, k)
        # print(f"\tCurrent moment is {moment}")
        w_new = w - delta * approx_gradient(w, samples, k)
        w_new = w_new / np.linalg.norm(w_new)
        # print(f"\tAttempt to move to {w_new}")
        new_moment = approx_moment(w_new, samples, k)
        # print(f"\tNew location's moment is {new_moment}")
        if new_moment >= moment:
            return w
        w = w_new
        iter += 1