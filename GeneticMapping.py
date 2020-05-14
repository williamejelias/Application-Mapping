import math
import random
import sys
import numpy as np
from matplotlib import pyplot as plt
import functools
import Allocation
import pickle
import CastNetMapping
import WCTG_Initialiser


class GeneticMapper:
    def __init__(self, file, pop_size, mutate_prob, cut_off, debug=False):
        print("***Genetic Mapping***")
        print("Reading in file...")
        self.num_nodes, self.node_weights, self.comm_matrix = WCTG_Initialiser.initialise_wctg_from_dataset_random(file)
        self.debug = debug
        print("Initialised WCTG.")

        ca = CastNetMapping.CastNetMapper(filename, debug)
        self.ca_maps = ca.map()

        # with open('../Results/CastNet256.pkl', 'rb') as f:
        #     self.ca_maps = pickle.load(f)

        self.fitness_sum = 0
        self.mutate_prob = mutate_prob
        self.pop_size = pop_size
        self.current_cycle = 0
        self.cut_off = cut_off

        self.best_cost = math.inf
        self.best_mapping = None
        self.best_allocation = None
        self.updated = False
        self.data = []

        self.debug = debug

    def initial_population(self, alloc_helper):
        # array of length size random mappings
        # each random mapping is a tuple of a mapping and a coordinate allocation dictionary
        return [alloc_helper.generate_random_mapping() for _ in range(self.pop_size)]

    def build_roulette_wheel(self, population):
        self.updated = False
        # generates a roulette based selection parameter
        # also updates the current best mapping

        # find inverse sum of fitnesses of mappings in population
        # a lower mapping cost is preferable
        self.fitness_sum = 0
        for m in population:
            mapping, allocation_dict, cost = m
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_mapping = mapping
                self.best_allocation = allocation_dict
                print("New best cost: ", self.best_cost)
                self.updated = True

                # reset current cycle
                self.current_cycle = 0
            self.fitness_sum += 1 / cost

    # use a roulette wheel to pick two parent mappings from the population
    # the greater the fitness of a mapping, the greater the probability of it becoming a parent
    def roulette_wheel(self, population):
        pick = random.uniform(0, self.fitness_sum)
        current = 0
        for m in population:
            mapping_cost = m[2]
            current += 1/mapping_cost
            if current >= pick:
                return m

    # generate a child mapping given two parent mappings and mutate given probability
    def crossover(self, parent1, parent2, alloc_helper):
        # each parent is a tuple of mapping and coordinate allocation dictionary
        p1_map, p2_map = parent1[0], parent2[0]
        mesh_shape = p1_map.shape

        p1_string, p2_string = p1_map.flatten(), p2_map.flatten()

        random_index = math.floor(int(random.uniform(0, len(p1_string))))
        parent1p1, parent1p2 = p1_string[random_index:], p1_string[:random_index]
        parent2p1, parent2p2 = p2_string[random_index:], p2_string[:random_index]
        child1 = np.concatenate((parent1p1, parent2p2))
        child2 = np.concatenate((parent2p1, parent1p2))

        # build list of repeated cities in each child
        temp_list1 = []
        temp_list2 = []
        repeated_values_child1 = []
        repeated_values_child2 = []
        for i in range(len(child1)):
            c1i = child1[i]
            c2i = child2[i]
            if c1i != 0:
                if c1i not in temp_list1:
                    temp_list1.append(c1i)
                else:
                    repeated_values_child1.append(c1i)

            if c2i != 0:
                if c2i not in temp_list2:
                    temp_list2.append(c2i)
                else:
                    repeated_values_child2.append(c2i)

        temp_list1 = []
        temp_list2 = []
        for i in range(len(child1)):
            if child1[i] not in temp_list1:
                temp_list1.append(child1[i])
            else:
                if len(repeated_values_child2) > 0:
                    child1[i] = repeated_values_child2.pop()
                else:
                    child1[i] = 0.0
            if child2[i] not in temp_list2:
                temp_list2.append(child2[i])
            else:
                if len(repeated_values_child1) > 0:
                    child2[i] = repeated_values_child1.pop()
                else:
                    child2[i] = 0.0

        # reshape new children
        c1_map = child1.reshape(mesh_shape)
        c2_map = child2.reshape(mesh_shape)

        # rebuild the allocation dict for each child mapping
        c1_alloc = alloc_helper.get_allocation_dict_from_mapping(c1_map)
        c2_alloc = alloc_helper.get_allocation_dict_from_mapping(c2_map)

        child1 = (c1_map, c1_alloc)
        child2 = (c2_map, c2_alloc)

        # find best child
        mutated1 = self.mutate_child(child1, alloc_helper)
        mutated2 = self.mutate_child(child2, alloc_helper)

        child1_cost = alloc_helper.total_comm_cost(mutated1[1])
        child2_cost = alloc_helper.total_comm_cost(mutated2[1])

        if child1_cost <= child2_cost:
            best_child = child1
            best_cost = child1_cost
        else:
            best_child = child2
            best_cost = child2_cost

        return (*best_child, best_cost)

    # perform a mutation on a mapping given a probability
    def mutate_child(self, child, alloc_helper):
        p = random.uniform(0, 1)
        if p < self.mutate_prob:
            mutated = alloc_helper.random_swap(*child)
            mutated_cost = alloc_helper.total_comm_cost(mutated[1])
            return (*mutated, mutated_cost)
        else:
            return child

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')

        ax.set_ylabel("Communication Costs", fontsize=14)
        ax.set_xlabel("Iterations", fontsize=14)
        label = "Genetic Mapping Algorithm - Population " + str(self.pop_size)
        plt.title(label)
        plt.plot(self.data, color='green')
        plt.show()
        # plt.savefig("../Results/result.png")

    def map(self):
        # create 2D square basic mesh and randomly allocate
        print("\nCreating initial population...")
        side_len = math.ceil(math.sqrt(self.num_nodes))
        mesh_dims = [side_len, side_len]
        num_tiles = functools.reduce(lambda x, y: (x * y), mesh_dims, 1)

        alloc_helper = Allocation.Allocation(self.num_nodes, self.comm_matrix, mesh_dims, self.node_weights)

        # generate initial population
        population = self.initial_population(alloc_helper)
        population += self.ca_maps
        self.pop_size += len(self.ca_maps)

        self.build_roulette_wheel(population)
        print("Population created.")

        print("\n\nEvolving population...")
        while self.current_cycle < num_tiles**2:
            # create new population
            new_population = []
            for i in range(self.pop_size):
                parent1 = self.roulette_wheel(population)
                parent2 = self.roulette_wheel(population)
                mutated_child = self.crossover(parent1, parent2, alloc_helper)
                new_population.append(mutated_child)
                population.append(mutated_child)

            population.sort(key=lambda x: x[2])

            # cull unfit members of population
            population = population[:self.pop_size]

            # build the new roulette wheel
            # also find the best tour/fitness of the current population
            self.build_roulette_wheel(population)
            self.data.append(self.best_cost)
            if not self.updated:
                self.current_cycle += 1

        print("\nBest mapping has cost: ", self.best_cost)
        print(self.best_mapping)


if __name__ == "__main__":
    try:
        filename = sys.argv[1]
        population_size = int(sys.argv[2])
        debug = False
        ga = GeneticMapper(filename, population_size, 0.35, 150, debug)
        ga.map()
        # ga.plot()
        # genetic_mapping(filename, population_size)
    except IndexError as msg:
        print(msg)
        exit()
