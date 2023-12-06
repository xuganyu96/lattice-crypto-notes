# GGH signature cryptanalysis
This is a Python implementation of the cryptanalysis described [here](https://cims.nyu.edu/~regev/papers/gghattack.pdf). To run the cryptanalysis on the command line:

```bash
python -m ggh
```

Or in Python

```python
from ggh.attack import simulate_attack

if __name__ == "__main__":
    true_sk, approx_sk, samples = simulate_attack(...)
```