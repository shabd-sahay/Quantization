"""Feature scaling for EMBER vectors. NumPy only - no torch, so the deployed app stays small."""
import numpy as np


class Scaler:
    """log1p on magnitudes, standardise, then clip.

    The clip is load-bearing, not cosmetic. EMBER features are counts and byte sizes with
    heavy tails; after log1p + standardisation the test set still reaches ~1000 sigma, because
    the statistics come from training rows. Dynamic int8 quantisation scales the activation
    tensor per-tensor, so those outliers set a scale that collapses every ordinary value to
    zero - which drove TPR@1%FPR from 0.789 to 0.178 before this clip existed. Clipping at
    +/-5 touches 0.08% of values and leaves fp32 detection unchanged.
    """

    def __init__(self, mean=None, std=None, clip=5.0):
        self.mean, self.std, self.clip = mean, std, clip

    @staticmethod
    def _t(X):
        return np.sign(X) * np.log1p(np.abs(X, dtype=np.float64))

    def fit(self, X):
        Z = self._t(X)
        self.mean = Z.mean(0)
        self.std = Z.std(0)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, X):
        Z = (self._t(X) - self.mean) / self.std
        if self.clip:
            Z = np.clip(Z, -self.clip, self.clip)
        return Z.astype(np.float32)

    def save(self, p):
        np.savez(p, mean=self.mean, std=self.std, clip=self.clip)

    @classmethod
    def load(cls, p):
        d = np.load(p)
        return cls(d["mean"], d["std"], float(d["clip"]) if "clip" in d else 5.0)
