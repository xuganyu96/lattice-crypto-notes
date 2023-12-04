# Let implement a GGH scheme first
import unittest
import numpy as np
from ggh.arithmetic import nearest_plane
from ggh.signature import keygen, sign, verify


class TestSignature(unittest.TestCase):
    repetition = 100
    mval_range = 99999

    def setUp(self) -> None:
        self.n = 5
        self.pk, self.sk, self.radius = keygen(self.n)

    def test_correctness(self):
        """Test that randomly generated message-signature pairs can be
        correctly verified
        """
        for _ in range(self.repetition):
            m = np.random.randint(-self.mval_range, self.mval_range, size=(self.n,))
            sigma = sign(m, self.sk)
            assert verify(m, sigma, self.pk, self.radius)
    
    def test_nearest_plane_resistance(self):
        """Test that attempting to use public key to forge signature does not
        work
        """
        for _ in range(self.repetition):
            m = np.random.randint(-self.mval_range, self.mval_range, size=(self.n,))
            forgery = self.pk * nearest_plane(m, self.pk)
            assert not verify(m, forgery, self.pk, self.radius)
