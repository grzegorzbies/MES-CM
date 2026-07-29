# MES-CM: Metaheuristic Exponential Search Optimization with Covariance Matrix Adaptation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![MEALPY](https://img.shields.io/badge/MEALPY-compatible-green.svg)](https://mealpy.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**MES-CM** is a single-solution metaheuristic optimization algorithm based on exponential pseudo-random sampling and covariance-guided search.

The method was designed as a non-metaphor-based optimizer that generates candidate solutions over multiple search scales. It combines short-range refinements with occasional longer exploratory moves and uses covariance information estimated from previously accepted solutions to guide the search direction.

---

## Table of contents

- [Overview](#overview)
- [Main features](#main-features)
- [Repository structure](#repository-structure)
- [Installation](#installation)
  - [Standalone installation](#standalone-installation)
  - [MEALPY installation](#mealpy-installation)
- [Usage with MEALPY](#usage-with-mealpy)
- [Standalone usage](#standalone-usage)
- [Parameters](#parameters)
  - [`MESCM`](#mescm)
  - [`MESCMStandalone`](#mescmstandalone)
- [Optimization log](#optimization-log)
- [Notes](#notes)
- [Authors](#authors)
- [License](#license)

---

## Overview

**Metaheuristic Exponential Search Optimization with Covariance Matrix Adaptation (MES-CM)** is a compact optimization method intended for continuous numerical optimization problems.

Unlike population-based metaheuristics, MES-CM operates with a single candidate solution. During the search process, the algorithm repeatedly generates a new candidate around the current solution. If the candidate improves the objective function value, it is accepted and becomes the new reference point for the next search step.

The main idea behind MES-CM is to use an exponential pseudo-random sampling mechanism to generate search steps over different distance scales. This allows the algorithm to perform both local exploitation and broader exploration without relying on a population of candidate solutions.

To improve directional search, MES-CM also uses a covariance-guided transformation. The covariance information is estimated from previously accepted solutions stored in a small observation archive. Once enough accepted solutions have been collected, this information is used to transform newly generated search steps according to the estimated local structure of the search space.

The repository provides two implementations in separate files:

| File | Class | Description |
|---|---|---|
| `MESCM.py` | `MESCM` | MEALPY-compatible optimizer |
| `MESCMStandalone.py` | `MESCMStandalone` | Standalone implementation with a simple Python interface |

Both implementations are intended for **minimization problems**.

The classes are separated so that `MESCMStandalone` can be used without installing MEALPY.

---

## Main features

- Single-solution optimization framework
- Exponential pseudo-random sampling mechanism
- Multi-scale candidate generation
- Covariance-guided transformation based on accepted solutions
- Optional target-fitness stopping condition
- MEALPY-compatible implementation
- Standalone implementation without MEALPY dependency
- Simple improvement log containing accepted objective function improvements

---

## Repository structure

```text
MES-CM/
├── MESCM.py              # MEALPY-compatible MES-CM implementation
├── MESCMStandalone.py    # Standalone MES-CM implementation without MEALPY
├── README.md            # Project documentation
├── LICENSE              # MIT license
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/grzegorzbies/MES-CM.git
cd MES-CM
```

### Standalone installation

If you want to use only `MESCMStandalone`, install only NumPy:

```bash
pip install numpy
```

This version does not require MEALPY.

### MEALPY installation

If you want to use the MEALPY-compatible `MESCM` class, install both NumPy and MEALPY:

```bash
pip install numpy mealpy
```

---

## Usage with MEALPY

Use this version if you want to run MES-CM through the MEALPY optimizer interface.

```python
import numpy as np
from mealpy import FloatVar
from MESCM import MESCM


def sphere(solution):
    solution = np.asarray(solution, dtype=float)
    return np.sum(solution ** 2)


problem = {
    "obj_func": sphere,
    "bounds": FloatVar(
        lb=(-10.0,) * 30,
        ub=(10.0,) * 30,
        name="x",
    ),
    "minmax": "min",
    "name": "Sphere",
    "log_to": "console",
}


model = MESCM(
    epoch=10000,
    pop_size=1,
    ft=1e-12,
    prec=30,
    sp=0.8,
    dp=0.35,
    obs=40,
)

g_best = model.solve(problem)

print("Best solution:", g_best.solution)
print("Best fitness:", g_best.target.fitness)
print("Improvement log:", model.get_log())
```

---

## Standalone usage

Use this version if you want to run MES-CM without MEALPY.

```python
import numpy as np
from MESCMStandalone import MESCMStandalone


def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return np.sum(
        100.0 * (x[1:] - x[:-1] ** 2) ** 2
        + (x[:-1] - 1.0) ** 2
    )


lb = [-10.0, -10.0, -10.0]
ub = [10.0, 10.0, 10.0]

model = MESCMStandalone(
    epoch=10000,
    seed=100,
    ft=1e-12,
    prec=30,
    sp=0.8,
    dp=0.35,
    obs=40,
)

result = model.solve(
    obj_func=rosenbrock,
    lb=lb,
    ub=ub,
    verbose=True,
)

print("Best fitness:", result["best_fitness"])
print("Best solution:", result["best_solution"])
print("Improvement log:", result["log"])
```

The standalone implementation solves minimization problems. To solve a maximization problem, convert it to minimization, for example:

```python
def objective_to_minimize(x):
    return -original_function_to_maximize(x)
```

---

## Parameters

### `MESCM`

| Parameter | Default | Description |
|---|---:|---|
| `epoch` | `50000` | Maximum number of iterations |
| `pop_size` | `1` | Population size required by the MEALPY interface; MES-CM is a single-solution method |
| `ft` | `-1e6` | Target fitness value used as an optional stopping condition |
| `prec` | `30` | Precision parameter used in the exponential sampling mechanism |
| `sp` | `0.8` | Search parameter controlling the use of standard-deviation-based scaling |
| `dp` | `0.35` | Direction parameter controlling the use of covariance-guided transformation |
| `obs` | `40` | Number of accepted solutions stored for covariance estimation |

### `MESCMStandalone`

| Parameter | Default | Description |
|---|---:|---|
| `epoch` | `50000` | Maximum number of iterations |
| `ft` | `-1e6` | Target fitness value used as an optional stopping condition |
| `prec` | `30` | Precision parameter used in the exponential sampling mechanism |
| `sp` | `0.8` | Search parameter controlling the use of standard-deviation-based scaling |
| `dp` | `0.35` | Direction parameter controlling the use of covariance-guided transformation |
| `obs` | `40` | Number of accepted solutions stored for covariance estimation |
| `seed` | `None` | Random seed used by the standalone implementation |

---

## Optimization log

Both implementations provide a log of accepted improvements.

For `MESCM`:

```python
log = model.get_log()
```

For `MESCMStandalone`:

```python
result = model.solve(obj_func, lb, ub)
log = result["log"]
```

The log contains tuples in the following format:

```text
(epoch, fitness)
```

where `epoch` is the iteration in which an improved candidate was accepted, and `fitness` is the corresponding objective function value.

---

## Notes

- MES-CM is a single-solution algorithm.
- The algorithm is intended for minimization problems.
- The MEALPY-compatible class uses `pop_size=1` by default.
- `MESCMStandalone` does not require MEALPY.
- The algorithm is not designed for parallel population evaluation.
- The covariance-guided mechanism is activated after enough accepted solutions have been collected.
- The standalone version returns a dictionary containing the best solution, best fitness, and optimization log.
- For reproducible experiments, several independent runs should be performed and reported.

---

## Authors

Created by:

- Grzegorz Bieś
- Ernest Bieś

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
