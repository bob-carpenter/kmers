from dataclasses import dataclass

import simulate_reads as sr
import transcriptome_reader as tr
from rna_seq_reader import reads_to_y
from solver_comparison.expconf import ExpConf
from solver_comparison.problem.model import (
    KmerModel,
    KmerModelName,
    KmerModels,
    model_classes,
)
from utils import Problem as KmerExprProblem
from utils import load_lengths


@dataclass
class Problem(ExpConf):
    """Wrapper around the datasets and model in kmerexpr."""

    model_name: KmerModelName
    filename: str
    K: int
    N: int
    L: int
    beta: float

    def load_model(self) -> KmerModel:
        """Creates data for the problem and loads the model -- Time hungry."""
        problem = KmerExprProblem(self.filename, K=self.K, N=self.N, L=self.L)
        (ISO_FILE, READS_FILE, X_FILE, Y_FILE) = problem.get_path_names()
        sr.simulate_reads(problem)
        reads_to_y(self.K, READS_FILE, Y_FILE=Y_FILE)
        tr.transcriptome_to_x(self.K, ISO_FILE, X_FILE, L=self.L)
        lengths = load_lengths(self.filename, self.N, self.L)

        return model_classes[self.model_name](
            X_FILE, Y_FILE, beta=self.beta, lengths=lengths, solver_name=None
        )


@dataclass
class NamedProblem(Problem):
    name: str


def simple(model_name: KmerModelName) -> NamedProblem:
    """Small test problem that can be solved in ~1s."""
    return NamedProblem(
        name=f"Simple{model_name}",
        filename="test5.fsa",
        K=8,
        N=1_000,
        L=14,
        beta=1.0,
        model_name=model_name,
    )


def SimpleLogistic() -> NamedProblem:
    return simple("Logistic")


def SimpleSimplex() -> NamedProblem:
    return simple("Simplex")


PROBLEMS = {p.name: p for p in [simple(_.__name__) for _ in KmerModels]}
"""All problems available for testing"""
