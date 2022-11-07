import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np
from numpy.typing import NDArray
from solver_comparison.expconf import ExpConf
from solver_comparison.problem.snapshot import Snapshot


@dataclass
class Optimizer(ABC, ExpConf):
    """Base class for optimizers."""

    max_iter: int = 100
    p_tol: float = 10 ** -16
    g_tol: float = 10 ** -16
    f_tol: float = 10 ** -16
    iter: int = field(init=False)

    def __post_init__(self):
        self.iter = 0

    @abstractmethod
    def step(self, current: Snapshot) -> Snapshot:
        pass

    def should_stop(self, df: float, dx: NDArray, dg: NDArray):
        i_check = self.iter > self.max_iter
        f_check = np.abs(df) <= self.f_tol
        p_check = np.linalg.norm(dx) ** 2 <= self.p_tol
        g_check = np.linalg.norm(dg) ** 2 <= self.g_tol
        return any([i_check, f_check, p_check, g_check])


@dataclass
class GDLS(Optimizer):
    """Gradient Descent using a Backtracking Armijo Linesearch.

    Looks for sufficient progress under Euclidean smoothness.

    Args:
        c: The strength of the Armijo condition, (0, 1).
           Larger values ask for more progress but is more difficult to satisfy.
        max: Maximum step-size allowed and starting point for backtracking.
        decr: Multiplicative factor when backtracking, (0, 1)
        incr: Multiplicative factor at the start of a new iteration [1, ...)
        max_iter: Maximum number of inner iterations
    """

    c: float = 0.5
    max: float = 10 ** 10
    decr: float = 0.5
    incr: float = 1.0
    max_iter: int = 100
    curr_ss: float = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        if not (0 < self.c < 1):
            raise ValueError(f"Strength parameter should be 0 < c < 1. Got {self.c}")
        if not (0 < self.decr < 1):
            raise ValueError(f"Mult. decr should be 0 < decr < 1. Got {self.decr}")
        if not (1.0 <= self.incr):
            raise ValueError(f"Mult. incr should be 1 <= incr. Got {self.incr}")
        if not isinstance(self.max_iter, int) or not (0 < self.max_iter):
            raise ValueError(f"max_iter should be an integer > 0. Got {self.incr}")
        self.curr_ss = self.max

    def step(self, current: Snapshot) -> Snapshot:
        def newpoint(ss: float) -> NDArray:
            return current.param - ss * current.g()

        logger = logging.getLogger(__name__)

        f_curr, g_curr = current.f(), current.g()
        curr_grad_norm = self.c * np.linalg.norm(g_curr) ** 2
        self.curr_ss = self.incr * self.curr_ss

        ss_s: List[float] = []
        f_s: List[float] = []
        found = False
        new = None
        for t in range(self.max_iter):
            new = Snapshot(param=newpoint(self.curr_ss), model=current.model)

            try:
                new.f()
            except FloatingPointError:
                logger.debug(
                    f"Overflow in linesearch with step-size {self.curr_ss:.2e} at "
                    f"parameter with norm {np.linalg.norm(new.p()):.2e}"
                )
                ss_s.append(self.curr_ss)
                f_s.append(np.nan)
                self.curr_ss = self.curr_ss * self.decr
                continue

            ss_s.append(self.curr_ss)
            f_s.append(new.f())

            sufficient_decrease = new.f() < f_curr - self.curr_ss * curr_grad_norm
            if sufficient_decrease:
                found = True
                break
            else:
                self.curr_ss = self.curr_ss * self.decr

        if not found:
            logger.warning(
                "LineSearch terminated without finding appropriate parameter."
            )
            return current

        assert new is not None
        return new
