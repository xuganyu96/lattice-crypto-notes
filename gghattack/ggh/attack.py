import time
import numpy as np
from sympy import Matrix
from .signature import SecretKey, generate_keypair, LatticeBasis
    

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
    # print(f"Generated {sample_size} message-signature pairs in {dur:.2f} seconds")

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

def simulate_attack(lattice_dim: int, sample_size: int):
    """Simulate a cryptanalysis. return the true secret basis and the 
    approximated/learned secret basis
    """
    sk = SecretKey.generate_sk(lattice_dim)
    sample_pairs = generate_pairs(sample_size, sk)
    samples = [(sigma - m) * 2 for (m, sigma) in sample_pairs]
    # sample_xs = [sample[0] for sample in samples]
    # sample_ys = [sample[1] for sample in samples]
    samples_sym = Matrix(np.array(samples).astype(int))
    approx_covariance = (
        samples_sym.transpose() * samples_sym / samples_sym.shape[0] * 3
    )
    approx_cholesky = approx_covariance.inv().cholesky(hermitian=False)
    cube_samples_sym = samples_sym * approx_cholesky
    cube_samples = np.array(cube_samples_sym.evalf()).astype(float)

    approx_sk = []
    for w_init in np.eye(lattice_dim):
        approx_cube_base = gradient_descent(cube_samples, 0.1, w_init)
        v = np.array(
            approx_cholesky.evalf().inv().transpose(), dtype=float
        ).dot(approx_cube_base)
        approx_sk.append(np.round(v))
    approx_sk = LatticeBasis(np.array(approx_sk).transpose())

    return sk, approx_sk