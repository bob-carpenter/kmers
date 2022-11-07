import base64
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from solver_comparison.expconf import ExpConf
from solver_comparison.log import (
    DataLogger,
    RateLimitedLogger,
    runtime,
    seconds_to_human_readable,
)
from solver_comparison.problem.problem import Problem
from solver_comparison.problem.snapshot import Snapshot
from solver_comparison.solvers.initializer import Initializer
from solver_comparison.solvers.optimizer import Optimizer


@dataclass
class Experiment(ExpConf):
    prob: Problem
    opt: Optimizer
    init: Initializer

    def hash(self):
        """A hash of the on ``uname`` encoded in b32 (filesystem-safe)."""
        as_bytes = self.uname().encode("ascii")
        as_hash = hashlib.sha256(as_bytes)
        as_b32 = base64.b32encode(as_hash.digest()).decode("ascii")
        return as_b32

    def uname(self):
        """A unique name that can be used to check for equivalence."""
        return base64.b32encode(self.as_str().encode("ascii")).decode("ascii")

    def _startup(self):
        self.datalogger = DataLogger(exp_id=self.hash(), exp_conf=self.as_dict())

        logger = logging.getLogger(__name__)
        logger.info("Initializing problem")

        with runtime() as loading_time:
            model = self.prob.load_model()

        self.datalogger.log({"model_load_time": loading_time.time})
        logger.info(f"Problem initialized in {loading_time.time:.2f}s")
        p = self.init.initialize_model(model)
        return Snapshot(model=model, param=p)

    @staticmethod
    def dummy_callback(*args, **kwargs):
        pass

    def run(
        self,
        progress_callback: Optional[
            Callable[["Experiment", float, Optional[Snapshot]], None]
        ],
    ):
        curr_p = self._startup()
        start_time = time.perf_counter()

        for t in range(self.opt.max_iter):

            with runtime() as iteration_time:
                new_p = self.opt.step(curr_p)

            df, dp, dg = (
                new_p.f() - curr_p.f(),
                new_p.p() - curr_p.p(),
                new_p.g() - curr_p.g(),
            )

            curr_time = time.perf_counter() - start_time
            self.datalogger.log(
                {
                    "time": curr_time,
                    "iter_time": iteration_time.time,
                    "f_before": curr_p.f(),
                    "f_after": new_p.f(),
                    "df": df,
                    "|dg|_2": np.linalg.norm(dg, ord=2),
                    "|dg|_1": np.linalg.norm(dg, ord=1),
                    "|dg|_inf": np.linalg.norm(dg, ord=np.inf),
                    "|dp|_2": np.linalg.norm(dp, ord=2),
                    "|dp|_1": np.linalg.norm(dp, ord=1),
                    "|dp|_inf": np.linalg.norm(dp, ord=np.inf),
                }
            )

            curr_p = new_p

            self.datalogger.end_step()
            if progress_callback is not None:
                progress_callback(self, t / self.opt.max_iter, curr_p)

            if self.opt.should_stop(df, dp, dg):
                break

        self.datalogger.save()


class ExperimentMonitor:
    def __init__(self, log_every: int = 3):
        self.timelogger = RateLimitedLogger(time_interval=log_every)
        self.start_time = time.perf_counter()

    def progress_callback(
        self, exp: Experiment, progress: float, snap: Optional[Snapshot]
    ):

        max_iter = exp.opt.max_iter
        i = int(max_iter * progress)
        i_width = len(str(max_iter))
        iter_str = f"Iter {i: >{i_width}}/{max_iter}"

        time_str = ""
        if self.start_time is not None:
            run_s = time.perf_counter() - self.start_time
            run_h = seconds_to_human_readable(run_s)

            eta_h, rem_h = "?", "?"
            if progress > 0:
                eta_s = run_s / progress
                rem_s = eta_s - run_s
                eta_h = seconds_to_human_readable(eta_s)
                rem_h = seconds_to_human_readable(rem_s)

            time_str = f"{run_h:>3}/{eta_h:>3} ({rem_h:>3} rem.)"

        data_str = ""
        if snap is not None:
            f, g = snap.f(), snap.g()
            data_str = f"Loss={f:.2e}  gnorm={np.linalg.norm(g):.2e}"

        self.timelogger.log(f"{iter_str } [{time_str:>18}] - {data_str}")
