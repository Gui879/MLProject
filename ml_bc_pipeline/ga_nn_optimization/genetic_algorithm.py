import numpy as np
from functools import reduce
from random_search import RandomSearch
from solutions.solution import Solution


class GeneticAlgorithm(RandomSearch):
    def __init__(self, problem_instance, random_state, population_size,
                 selection, crossover, p_c, mutation, p_m):
        RandomSearch.__init__(self, problem_instance, random_state)
        self.population_size = population_size
        self.selection = selection
        self.crossover = crossover
        self.p_c = p_c
        self.mutation = mutation
        self.p_m = p_m

    def initialize(self):
        self.population = self._generate_random_valid_solutions()
        self.best_solution = self._get_elite(self.population)

    def search(self, n_iterations, report=False):
        elite = self.best_solution

        for iteration in range(n_iterations):
            offsprings = []

            while len(offsprings) < len(self.population):
                off1, off2 = p1, p2 = [
                    self.selection(self.population, self.problem_instance.minimization, self._random_state)
                    for i in range(2)]

                if self._random_state.uniform() < self.p_c:
                    off1, off2 = self._crossover(p1, p2)

                if self._random_state.uniform() < self.p_m:
                    off1 = self._mutation(off1)
                    off2 = self._mutation(off2)

                self.problem_instance.evaluate(off1)
                self.problem_instance.evaluate(off2)
                offsprings.extend([off1, off2])

            while len(offsprings) > len(self.population):
                offsprings.pop()

            self.population = offsprings
            elite = self._get_elite(self.population)

            if report:
                self._verbose_reporter_inner(elite, iteration)

        self.best_solution = elite

    def _crossover(self, p1, p2):
        off1, off2 = self.crossover(p1.representation, p2.representation, self._random_state)
        off1, off2 = Solution(off1), Solution(off2)
        return off1, off2

    def _mutation(self, individual):
        mutant = self.mutation(individual.representation, self._random_state)
        mutant = Solution(mutant)
        return mutant

    def _get_elite(self, population):
        elite = reduce(self._get_best, population)
        return elite

    def _generate_random_valid_solutions(self):
        solutions = np.array([self._generate_random_valid_solution()
                              for i in range(self.population_size)])
        return solutions
