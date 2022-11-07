import numpy as np
from solver_comparison.experiment import Experiment, ExperimentMonitor
from solver_comparison.problem.problem import SimpleLogistic
from solver_comparison.solvers.initializer import Initializer
from solver_comparison.solvers.optimizer import GDLS

if __name__ == "__main__":

    np.seterr(all="raise")

    experiments = [
        Experiment(
            prob=prob,
            opt=GDLS(max=1.0, incr=1.1, max_iter=100),
            init=Initializer("zero"),
        )
        for prob in [SimpleLogistic()]
    ]

    for exp in experiments:
        exp.run(progress_callback=ExperimentMonitor(log_every=1).progress_callback)
