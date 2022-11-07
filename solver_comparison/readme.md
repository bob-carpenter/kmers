# Solver comparison

This codebase wraps `kmerexpr` to write a small benchmarking library.

Main files/concepts: 

**Problem definitions**
- [`Model`](src/solver_comparison/problem/model.py) 
  Definition of the loss function and gradient given a dataset.  
  Wraps the code in
  [`multinomial_model`](../multinomial_model.py),
  [`multinomial_simplex_model`](../multinomial_simplex_model.py) 
  and [`normal_model`](../normal_model.py).
- [`Problem`](src/solver_comparison/problem/problem.py): Combination of a `Model` and a dataset.   
  Wraps the code in
  [`simulate_reads`](../simulate_reads.py) 
  [`transcriptome_reader`](../transcriptome_reader.py) 
  [`rna_seq_reader`](../rna_seq_reader.py) 

**Solvers**
- [`Initializer`](src/solver_comparison/solvers/initializer.py): Initialization strategies
- [`Optimizer`](src/solver_comparison/solvers/optimizer.py): Provide a generic interface to different opt 

**Benchmarking/Running**
- [`Experiment`](src/solver_comparison/experiment.py): Wraps a `Problem`, `Initializer`, `Optimizer`.  
  Runs an experiment and logs the results with some help 
- from other functions in [`log.py`](src/solver_comparison/log.py) (format to log) 
  and [`config.py`](src/solver_comparison/config.py) (where to log)

**Data logging**
- [`GlobalLogger.DataLogger`](src/solver_comparison/log.py): 
  Saving arbitrary data to a `.csv` depending on the experiment hash. 

**Missing** 
- Moving other optimizers and problems in this format (currently only plain GD)
- Managing experiment data when finished running (aka making plots)

---

**Additional dependencies:**
```
pandas
```

 
**Coding style**  
This code is formatted with 
[Black](https://github.com/psf/black) and 
[isort](https://github.com/PyCQA/isort).
Docstrings in 
[Google style](https://google.github.io/styleguide/pyguide.html#s3.8.1-comments-in-doc-strings)
are checked with
[docformatter](https://github.com/PyCQA/docformatter).

---

**Configuration:**

Where to save run data is specified through environment variable.  
Would eventually store additional info, for example to log results to wandb.  
Can be configured with an environment file, e.g.

- On OSX/Unix, `source env.sh` with `env.sh` containing
  ```
  export KMEREXPR_BENCH_WORKSPACE=~/path/to/workspace 
  ```
- On Windows, `call env.bat` with `env.bat` containing
  ```
  set KMEREXPR_BENCH_WORKSPACE=C:\User\user\path\to\workspace
  ```
