import argparse
from ggh.attack import simulate_attack

parser = argparse.ArgumentParser(
    prog="GGH Cryptanalysis",
    description="Cryptanalysis of the GGH signature scheme",
)
parser.add_argument("--dim", type=int, help="Dimension of the lattice", default=2)
parser.add_argument(
    "--samples",
    type=int,
    default=1000,
    help="Number of message-signature samples to generate",
)
parser.add_argument(
    "--descent-delta", type=float, default=0.7, help="Gradient descent speed"
)
parser.add_argument(
    "--magnitude",
    type=float,
    default=100,
    help="""Magnitude of the secret basis. Defaults to 100""",
)

if __name__ == "__main__":
    args = parser.parse_args()
    sk, approx_sk, _ = simulate_attack(
        args.dim, args.samples, args.descent_delta, args.magnitude
    )

    print(sk.dist(approx_sk))
