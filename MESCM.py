"""MES-CM optimizers.

Metaheuristic Exponential Search Optimization with Covariance Matrix adaptation.
"""

import math

import numpy as np
from mealpy.optimizer import Optimizer


ALGORITHM_NAME = "MES-CM"
ALGORITHM_DESCRIPTION = (
    "Metaheuristic Exponential Search Optimization with Covariance Matrix "
    "adaptation (MES-CM)"
)


class MESCM(Optimizer):
    def __init__(
        self,
        epoch=50000,
        pop_size=1,
        ft=-1e6,
        prec=30,
        sp=0.8,
        dp=0.35,
        obs=40,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.is_parallelizable = False

        self.epoch = epoch
        self.pop_size = pop_size if pop_size is not None else 1

        self.ftarget = ft
        self.searchParameter = sp
        self.directionParameter = dp
        self.observations = obs

        self.precision = prec

    # ================= INIT =================

    def initialize_variables(self):
        self._ranges = []
        self._lb = None
        self._ub = None
        self._denominators = []
        self._values = None
        self._standardValues = None
        self._std = None

        self._data = None
        self._index = 0
        self._dataReady = False

        self._generatedValuesCount = 0
        self._sqrtGenerationsCount = 0

        self.is_parallelizable = False

        self._logs = []

    def initialization(self):
        self.pop = self.generate_population(self.pop_size)

        best = self.pop[0]

        self._lb = np.array(self.problem.lb)
        self._ub = np.array(self.problem.ub)

        self._values = best.solution.copy()
        dim = len(self._values)

        self._data = np.zeros((self.observations, dim))
        self._sqrtGenerationsCount = math.sqrt(dim) * 2

        self._ranges = []
        self._denominators = []
        self._standardValues = []

        for i in range(dim):
            span = self._ub[i] - self._lb[i]
            self._ranges.append(int(span * (10**self.precision)))
            self._denominators.append(span)
            self._standardValues.append((self._values[i] - self._lb[i]) / span)

        self._ranges = np.array(self._ranges)
        self._denominators = np.array(self._denominators)
        self._standardValues = np.array(self._standardValues)

    # ================= EVOLVE =================

    def evolve(self, epoch):
        best = self.pop[0]

        candidate_standard = self.getCandidateStandardValue()
        candidate_values = self.convertStandardValuesToValues(candidate_standard)

        fitness = self.get_target(candidate_values).fitness

        if fitness < best.target.fitness:
            self.setValuesFromCandidate(candidate_standard)
            self.pop[0] = self.generate_agent(candidate_values)

            self._logs.append((epoch, fitness))

        # optional stop
        if best.target.fitness <= self.ftarget:
            self.termination_flag = True

    # ================= CORE =================

    def _checkValues(self, dValues):
        newValues = []

        for i, dValue in enumerate(dValues):
            val = self._standardValues[i] + dValue

            if val > 1.0:
                dValue -= 1
            elif val < 0.0:
                dValue += 1

            val = self._standardValues[i] + dValue

            if val > 1.0 or val < 0.0:
                dValue = 0.0

            newValues.append(dValue)

        return np.array(newValues)

    def _getDValues(self):
        dValues = []
        directivity = self.generator.random()
        Ux = self.generator.random()

        for i in range(len(self._values)):
            dValue = self.generateFromLevel(i, Ux)

            if self._dataReady and directivity > self.directionParameter:
                if self.generator.random() <= self.searchParameter:
                    dValue *= self._std[i]

            if self.generator.random() < 0.5:
                dValue = -dValue

            dValues.append(dValue)

        dValues = np.array(dValues)

        if self._dataReady and directivity <= self.directionParameter:
            try:
                dValues = np.matmul(self._L, dValues)
            except Exception:
                self._dataReady = False

        return dValues

    def getCandidateStandardValue(self):
        dValues = self._checkValues(self._getDValues())
        candidate = self._standardValues + dValues
        self._generatedValuesCount += 1
        return candidate

    def convertStandardValuesToValues(self, standardValues):
        return self._lb + standardValues * self._denominators

    def setValuesFromCandidate(self, candidate):
        self._values = self.convertStandardValuesToValues(candidate)
        self._standardValues = candidate

        if self._generatedValuesCount > self._sqrtGenerationsCount:
            self.addData(candidate)

        if self._dataReady and self._index == 0:
            self._std = np.std(self._data, axis=0)
            C = np.cov(self._data.T)

            try:
                self._L = np.linalg.cholesky(C + 1e-12 * np.eye(C.shape[0]))
            except Exception:
                self._dataReady = False

        self._generatedValuesCount = 0

    def generateFromLevel(self, i, level):
        Ux = level + (1.0 - level) * self.generator.random()
        return self._ranges[i] ** (Ux - 1.0)

    def addData(self, data):
        self._data[self._index] = data
        self._index += 1

        if self._index == self.observations:
            self._index = 0
            self._dataReady = True

    def get_name(self):
        return ALGORITHM_NAME

    def get_extra_info(self):
        return ALGORITHM_DESCRIPTION

    def get_log(self):
        return self._logs


# Standalone version without MEALPY

class MESCMStandalone:
    def __init__(
        self,
        epoch=50000,
        ft=-1e6,
        prec=30,
        sp=0.8,
        dp=0.35,
        obs=40,
        seed=None,
    ):
        self.epoch = epoch
        self.ftarget = ft

        self.precision = prec
        self.searchParameter = sp
        self.directionParameter = dp
        self.observations = obs

        self.seed = seed
        self.generator = np.random.default_rng(seed)

        self._ranges = None
        self._lb = None
        self._ub = None
        self._denominators = None

        self._values = None
        self._standardValues = None
        self._std = None
        self._L = None

        self._data = None
        self._index = 0
        self._dataReady = False

        self._generatedValuesCount = 0
        self._sqrtGenerationsCount = 0

        self._logs = []

        self.best_solution = None
        self.best_fitness = None

    # ================= INIT =================

    def _initialize(self, obj_func, lb, ub, start_solution=None):
        self.obj_func = obj_func

        self._lb = np.asarray(lb, dtype=float)
        self._ub = np.asarray(ub, dtype=float)

        if self._lb.shape != self._ub.shape:
            raise ValueError("lb and ub must have the same shape.")

        dim = len(self._lb)

        if start_solution is None:
            self._values = self.generator.uniform(self._lb, self._ub)
        else:
            self._values = np.asarray(start_solution, dtype=float)
            self._values = np.clip(self._values, self._lb, self._ub)

        self.best_solution = self._values.copy()
        self.best_fitness = self._evaluate(self.best_solution)

        self._data = np.zeros((self.observations, dim))
        self._sqrtGenerationsCount = math.sqrt(dim) * 2

        self._ranges = []
        self._denominators = []
        self._standardValues = []

        for i in range(dim):
            span = self._ub[i] - self._lb[i]

            if span <= 0:
                raise ValueError(f"Invalid bounds for variable {i}: ub <= lb.")

            self._ranges.append(int(span * (10**self.precision)))
            self._denominators.append(span)
            self._standardValues.append((self._values[i] - self._lb[i]) / span)

        self._ranges = np.array(self._ranges, dtype=float)
        self._denominators = np.array(self._denominators, dtype=float)
        self._standardValues = np.array(self._standardValues, dtype=float)

        self._std = None
        self._L = None

        self._index = 0
        self._dataReady = False
        self._generatedValuesCount = 0
        self._logs = []

    # ================= SOLVE =================

    def solve(self, obj_func, lb, ub, start_solution=None, verbose=False):
        self._initialize(obj_func, lb, ub, start_solution)

        for epoch in range(1, self.epoch + 1):
            candidate_standard = self.getCandidateStandardValue()
            candidate_values = self.convertStandardValuesToValues(candidate_standard)

            fitness = self._evaluate(candidate_values)

            if fitness < self.best_fitness:
                self.setValuesFromCandidate(candidate_standard)

                self.best_solution = candidate_values.copy()
                self.best_fitness = fitness

                self._logs.append((epoch, fitness))

                if verbose:
                    print(f"Epoch: {epoch}, Best fitness: {fitness}")

            if self.best_fitness <= self.ftarget:
                break

        return {
            "best_solution": self.best_solution.copy(),
            "best_fitness": self.best_fitness,
            "log": self._logs,
        }

    # ================= FITNESS =================

    def _evaluate(self, x):
        return float(self.obj_func(np.asarray(x, dtype=float)))

    # ================= CORE =================

    def _checkValues(self, dValues):
        newValues = []

        for i, dValue in enumerate(dValues):
            val = self._standardValues[i] + dValue

            if val > 1.0:
                dValue -= 1.0
            elif val < 0.0:
                dValue += 1.0

            val = self._standardValues[i] + dValue

            if val > 1.0 or val < 0.0:
                dValue = 0.0

            newValues.append(dValue)

        return np.array(newValues, dtype=float)

    def _getDValues(self):
        dValues = []

        directivity = self.generator.random()
        Ux = self.generator.random()

        for i in range(len(self._values)):
            dValue = self.generateFromLevel(i, Ux)

            if self._dataReady and directivity > self.directionParameter:
                if self.generator.random() <= self.searchParameter:
                    dValue *= self._std[i]

            if self.generator.random() < 0.5:
                dValue = -dValue

            dValues.append(dValue)

        dValues = np.array(dValues, dtype=float)

        if self._dataReady and directivity <= self.directionParameter:
            try:
                dValues = np.matmul(self._L, dValues)
            except Exception:
                self._dataReady = False

        return dValues

    def getCandidateStandardValue(self):
        dValues = self._checkValues(self._getDValues())

        candidate = self._standardValues + dValues

        self._generatedValuesCount += 1

        return candidate

    def convertStandardValuesToValues(self, standardValues):
        standardValues = np.asarray(standardValues, dtype=float)
        return self._lb + standardValues * self._denominators

    def setValuesFromCandidate(self, candidate):
        self._values = self.convertStandardValuesToValues(candidate)
        self._standardValues = np.asarray(candidate, dtype=float)

        if self._generatedValuesCount > self._sqrtGenerationsCount:
            self.addData(candidate)

        if self._dataReady and self._index == 0:
            self._std = np.std(self._data, axis=0)

            C = np.cov(self._data.T)

            try:
                self._L = np.linalg.cholesky(C + 1e-12 * np.eye(C.shape[0]))
            except Exception:
                self._dataReady = False

        self._generatedValuesCount = 0

    def generateFromLevel(self, i, level):
        Ux = level + (1.0 - level) * self.generator.random()
        return self._ranges[i] ** (Ux - 1.0)

    def addData(self, data):
        self._data[self._index] = data
        self._index += 1

        if self._index == self.observations:
            self._index = 0
            self._dataReady = True

    # ================= INFO =================

    def get_name(self):
        return ALGORITHM_NAME

    def get_extra_info(self):
        return ALGORITHM_DESCRIPTION

    def get_log(self):
        return self._logs

    def get_best_solution(self):
        if self.best_solution is None:
            return None

        return self.best_solution.copy()

    def get_best_fitness(self):
        return self.best_fitness