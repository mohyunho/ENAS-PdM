#/usr/bin/python3
"""
This file contains implementations of evolutionary algorithms to evolve neural networks in the context of predictive mainteinance.
Author: Leonardo Lucio Custode
Date: 17/09/2020
"""
import random
import pathos
import numpy as np
from task import Task
from functools import partial
from deap import base, algorithms, creator, tools
import pickle
import os
import pandas as pd
import copy

# os.remove("logbook.pkl")


class ListWithParents(list):
    def __init__(self, *iterable):
        super(ListWithParents, self).__init__(*iterable)
        self.parents = []


def varAnd(population, toolbox, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* because of its propensity to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematically, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]
    unmodified = [*range(len(offspring))]

    for i, o in enumerate(offspring):
        o.parents = [i]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            offspring[i-1].parents.append(i)
            offspring[i].parents.append(i - 1)
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            if i in unmodified:
                unmodified.remove(i)
            if i+1 in unmodified:
                unmodified.remove(i+1)

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            if i in unmodified:
                unmodified.remove(i)

    return offspring, unmodified


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, log_function=None):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])


    individual_map = {}

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        individual_map[str(ind)] = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring, unmodified = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        to_evaluate = []
        redundant = []
        for ind in invalid_ind:
            key = str(ind)
            if key in individual_map:
                ind.fitness.values = individual_map[key]
                redundant.append(ind)
            else:
                to_evaluate.append(ind)
        invalid_ind = to_evaluate

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            individual_map[str(ind)] = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        for o in offspring:
            argmin = np.argmin(map(lambda x: population[x].fitness.values[0], o.parents))

            if o.fitness.values[0] < population[o.parents[argmin]].fitness.values[0]:
                population[o.parents[argmin]] = o


        population_temp = copy.deepcopy(population)
        log_function(population_temp, gen)
        # not_mutated = [population_temp[u] for u in unmodified]
        # if len(unmodified) > 0 and log_function is not None:
        #     # print ([population_temp[u] for u in unmodified])
        #     log_function(not_mutated, gen)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}

        temp_list = []
        print ("population", population)
        for i in range(len(population)):
            # print("population.fitness.values", population[i].fitness.values)
            temp_list.append(population[i].fitness.values[0])
        print (temp_list)
        max_value = max(temp_list)
        min_value = min(temp_list)
        avg_value = 0 if len(temp_list) == 0 else sum(temp_list) / len(temp_list)
        print ("min: %s, max:%s, avg:%s"  %(min_value, max_value, avg_value))



        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)


    # with open("EA_log/logbook.pkl", "w") as lb_file:
    #     pickle.dump(logbook, lb_file)

    print ("pickle dump")
    pickle.dump(logbook, open("EA_log/logbook.pkl", "wb"))
    print ("log saved")
    return population, logbook


def checkBounds(bounds):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > bounds[i][1]:
                        child[i] = bounds[i][1]
                    elif child[i] < bounds[i][0]:
                        child[i] = bounds[i][0]
            return offspring
        return wrapper
    return decorator


class GeneticAlgorithm:
    def __init__(self, task: Task, population_size: int, n_generations: int, cx_probability: float, mut_probability: float, crossover_operator: str = "one_point", mutation_operator: str = "uniform", selection_operator: str = "best", seed=None, jobs=1, log_function=None, **kwargs):
        """
        Initializes an instance of the genetic algorithm.
        Parameters:
            - task: an instance of the class Task
            - population_size: the number of individuals used at each generation.
            - n_generations: the number of generations
            - cx_probability: probability that the crossover operator is applied to a couple of individuals
            - mut_probability: probability that the mutation operator is applied to an individual
            - crossover_operator: the operator used for crossover. Currently supporting:
                - one_point: one-point crossover
                - two_points: two-points crossover
                - uniform: uniform crossover. It requires the following parameters:
                    - cx_gene_probability: the probability of exchanging genes between individuals (i.e. the probability that the child of a parent exchanges genes with the other parent)
            - mutation_operator: the operator used for mutation. Currently supporting:
                - uniform: uniform mutation in the range of representation for the individuals. It requires the following parameters:
                    - mut_gene_probability: the probability of applying the mutation operator to a single gene
                - shuffle: shuffle indexes in the individual. It requires the following parameters:
                    - mut_gene_probability: the probability of applying the mutation operator to a single gene
            - selection_operator: the operator used for selection. Currently supporting:
                - best: select best individuals
                - tournament: tournament selection. Requires the following parameters:
                    - sel_tournament_size: integer
            - seed: a seed for the evolution (i.e. an individual that is a good starting point).
            - jobs: Number of jobs to use for the parallelization of the evolution
            - log_function: A function pointer to a logging function to log the individuals that are not mated/mutated
        """
        self._define_supported_operators()

        assert mutation_operator in self.supported_mutations,\
            "The mutation operator {} is not supported. Supported operators:\n\t{}".format(mutation_operator, list(self.supported_mutations.keys()))
        assert self.supported_mutations[mutation_operator] is None or \
            self.supported_mutations[mutation_operator] in kwargs,\
            "The selected mutation operator ({}) requires the following keyword parameter: {}".format(mutation_operator, self.supported_mutations[mutation_operator])
        assert crossover_operator in self.supported_crossovers,\
            "The crossover_operator {} is not supported. Supported operators:\n\t{}".format(crossover_operator, list(self.supported_crossovers.keys()))
        assert self.supported_crossovers[crossover_operator] is None or \
            self.supported_crossovers[crossover_operator] in kwargs,\
            "The selected crossover operator ({}) requires the following keyword parameter: {}".format(crossover_operator, self.supported_crossovers[crossover_operator])
        assert selection_operator in self.supported_selections,\
            "The selection operator {} is not supported. Supported operators:\n\t{}".format(selection_operator, list(self.supported_selections.keys()))
        assert self.supported_selections[selection_operator] is None or \
            self.supported_selections[selection_operator] in kwargs,\
            "The chosen selection operator ({}) requires the following keyword parameter: {}".format(selection_operator, self.supported_selections[selection_operator])

        self.task = task
        self.n_parameters = task.get_n_parameters()
        self.parameter_bounds = task.get_parameters_bounds()
        self.population_size = population_size
        self.n_generations = n_generations
        self.cx_probability = cx_probability
        self.mut_probability = mut_probability
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        self.selection_operator = selection_operator
        self.seed = seed
        self.jobs = jobs
        self.kwargs = kwargs
        self.log_function = log_function

        self._initialize_deap()

    def _define_supported_operators(self):
        self.supported_mutations = {"uniform": "mut_gene_probability", "shuffle": "mut_gene_probability"}
        self.supported_crossovers = {"one_point": None, "two_points": None, "uniform": "cx_gene_probability"}
        self.supported_selections = {"best": None, "tournament": "sel_tournament_size"}

    def _get_mutation_operator(self):
        mutate = None
        arg = None
        if self.mutation_operator == "uniform":
            mutate = tools.mutUniformInt
            arg = {"indpb": self.kwargs["mut_gene_probability"], "low": np.min(self.parameter_bounds), "up": np.max(self.parameter_bounds)}
        elif self.mutation_operator == "shuffle":
            mutate = tools.mutShuffleIndexes
            arg = {"indpb": self.kwargs["mut_gene_probability"]}
        return mutate, arg

    def _get_crossover_operator(self):
        mate = None
        arg = None
        if self.crossover_operator == "one_point":
            mate = tools.cxOnePoint
        elif self.crossover_operator == "two_points":
            mate = tools.cxTwoPoint
        else:
            mate = tools.cxUniform
            arg = {"indpb": self.kwargs["cx_gene_probability"]}
        return mate, arg

    def _get_selection_operator(self):
        sel = None
        arg = None
        if self.selection_operator == "best":
            sel = tools.selBest
        else:
            sel = tools.selTournament
            arg = {"tournament_size": self.kwargs["sel_tournament_size"]}
        return sel, arg

    def _register_operator(self, name, op, arg):
        if arg is None:
            self.toolbox.register(name, op)
        else:
            self.toolbox.register(name, op, **arg)

    def _initialize_deap(self):
        """
        This method sets up the required components of the DEAP library
        """

        # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # creator.create("IndiviIndividual", list, fitness=creator.FitnessMax)

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.creator = creator

        self.toolbox = base.Toolbox()

        if self.jobs > 1:
            pool = pathos.multiprocessing.Pool(processes=self.jobs)
            self.toolbox.register("map", pool.map)

        attributes = []
        for i, (min_, max_) in enumerate(self.parameter_bounds):
            self.toolbox.register("attr_{}".format(i), random.randint, min_, max_)
            attributes.append(eval("self.toolbox.attr_{}".format(i)))

        self.toolbox.register("individual", tools.initCycle, creator.Individual, tuple(attributes), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        operators = [("mate", *self._get_crossover_operator())]
        operators.append(("mutate", *self._get_mutation_operator()))
        operators.append(("select", *self._get_selection_operator()))

        for name, op, arg in operators:
            self._register_operator(name, op, arg)

        self.toolbox.decorate("mate", checkBounds(self.parameter_bounds))
        self.toolbox.decorate("mutate", checkBounds(self.parameter_bounds))

    def run(self):
        """
        Runs the optimization process.
        Parameters:
            - task: an instance of the "Task" class.
        Returns:
            - pop: the final population, a list of genotypes
            - log: the log of the evolution, with the statistics
            - hof: the hall of fame, containing the best individual
        """
        self.toolbox.register("evaluate", self.task.evaluate)
        pop = self.toolbox.population(n=self.population_size)
        if self.seed is not None:
            if isinstance(self.seed[0], int):
                pop[0] = self.creator.Individual(self.seed)
            else:
                assert isinstance(self.seed[0], list), "Seed must be a list of integers or a list of lists"
                for pop_idx, individual in enumerate(self.seed):
                    pop[pop_idx] = self.creator.Individual(individual)
                    if pop_idx == len(pop) - 1:
                        break

        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)


        pop, log = eaSimple(
            pop,
            self.toolbox,
            cxpb=self.cx_probability,
            mutpb=self.mut_probability,
            ngen=self.n_generations,
            stats=stats,
            halloffame=hof,
            verbose=True,
            log_function=self.log_function
        )

        return pop, log, hof
