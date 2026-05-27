# MES-CM optimization algorithm
A new optimization algorithm, called Metaheuristic Exponential Search Optimization with Covariance Matrix adaptation (MES-CM).

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Usage](#usage)
* [Contact](#contact)

## General info
Metaheuristic Exponential Search Optimization with Covariance Matrix adaptation (MES-CM) is a new optimization algorithm that, unlike most known population-based algorithms, uses only a single candidate solution throughout the optimization process. The operation of MES-CM is based on a proposed pseudorandom number generator, which is used to determine new values ​​for decision variables in subsequent iterations of the algorithm. Additionally, the algorithm adapts the covariance matrix, ensuring that each subsequent iteration uses a proposed solution (candidate) that is potentially better suited to the environment than previous iterations.

## Technologies
* Python

## Usage


```
from MESCM import MESCMStandalone
import numpy as np

def sphere(x):
    return np.sum(x ** 2)


def rosenbrock(x):
    x = np.asarray(x, dtype=float)

    return np.sum(
        100.0 * (x[:-1] ** 2 - x[1:]) ** 2
        + (x[:-1] - 1.0) ** 2
    )

lb = [-10, -10, -10]
ub = [10, 10, 10]

model = MESCMStandalone(
    epoch=10000,
    seed=100,
    ft=1e-12,
    minmax="min"
)

result = model.solve(
    #obj_func=sphere,
    obj_func=rosenbrock,
    lb=lb,
    ub=ub,
    verbose=True
)

print("Best fitness:", result["best_fitness"])
print("Best solution:", result["best_solution"])

```

## Contact
Created by Grzegorz Bieś [grzegorzbies75@gmail.com] & Ernest Bieś [ernestbies@gmail.com]
