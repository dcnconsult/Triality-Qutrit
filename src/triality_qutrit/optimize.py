from __future__ import annotations
import numpy as np

# Lightweight placeholder for Bayesian/bandit loop (hardware-in-the-loop possible).
# In practice, swap with BoTorch/GPyTorch models for acquisition-driven sampling.
class SimpleBandit:
    def __init__(self, xs, ys):
        self.points = [(x,y) for x in xs for y in ys]
        self.measured = {}
    def suggest(self, k=5):
        return [p for p in self.points if p not in self.measured][:k]
    def observe(self, xy, value):
        self.measured[xy] = value
