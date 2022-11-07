from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from solver_comparison.expconf import ExpConf
from solver_comparison.problem.model import KmerModel


@dataclass
class Initializer(ExpConf):
    _METHOD_TYPE = Literal["simplex_uniform", "zero"]
    method: _METHOD_TYPE = "simplex_uniform"

    def initialize_model(self, model: KmerModel) -> NDArray:
        if self.method == "simplex_uniform":
            return np.ones(model.T) / model.T
        elif self.method == "zero":
            return np.zeros(model.T)
