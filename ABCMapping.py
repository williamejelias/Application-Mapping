import functools
import math
import random
import sys

import Allocation
import CastNetMapping
import WCTG_Initialiser


class ABCMapper:
    def __init__(self, filename, sn, mcn, debug):
        print("***ABC Mapping***")
        print("Reading in file...")
        self.num_nodes, self.node_weights, self.comm_matrix = WCTG_Initialiser.initialise_wctg_from_dataset_random(filename)
        print("Initialised WCTG.")

        # allocation helper
        self.side_len = math.ceil(math.sqrt(self.num_nodes))
        self.mesh_dims = [self.side_len, self.side_len]
        self.num_tiles = functools.reduce(lambda x, y: (x * y), self.mesh_dims, 1)
        self.alloc_helper = Allocation.Allocation(self.num_nodes, self.comm_matrix, self.mesh_dims, self.node_weights)

        # ca = CastNetMapping.CastNetMapper(filename, debug)
        # self.ca_maps = ca.map()
        # self.ca_maps = math.floor(self.num_tiles / len(self.ca_maps)) * self.ca_maps
        # l = len(self.ca_maps)

        # set parameters
        self.maximum_cycle_number = int(mcn)
        self.population_size = int(sn)
        self.solution_improvement_limit = 5000
        self.swaps_per_bee = 10

        self.population = self.initial_population()

        self.best_mapping = None
        self.best_allocation = None
        self.best_cost = math.inf

        self.fitness_sum = math.inf
        self.updated = False

        self.current_cycle = 0

        # debug
        self.debug = debug

    def initial_population(self):
        # castnet_population = [[*i, 0, 0] for i in self.ca_maps]
        population = [[*self.alloc_helper.generate_random_mapping(), 0, 0] for _ in range(self.population_size)]
        # return population + castnet_population
        return population

    def build_roulette_wheel(self):
        self.updated = False
        # generates a roulette based selection parameter
        # also updates the current best mapping

        # find inverse sum of fitnesses of mappings in population
        # a lower mapping cost is preferable
        # self.mappings_costed = []
        self.fitness_sum = 0
        for m in self.population:
            # print(m)
            mapping, allocation_dict, cost = m[:-2]
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_mapping = mapping
                self.best_allocation = allocation_dict
                print("New best cost: ", self.best_cost)
                self.updated = True

                # reset current cycle
                self.current_cycle = 0
            self.fitness_sum += 1 / cost

    # the greater the fitness of a mapping, the greater the probability of it being chosen
    # returns the index of the mapping that was chosen
    def roulette_wheel(self):
        pick = random.uniform(0, self.fitness_sum)
        current = 0
        for m in range(len(self.population)):
            mapping_cost = self.population[m][2]
            current += 1 / mapping_cost
            if current >= pick:
                return m

    def map(self):
        # vars: col_size, cycle_improv_limit, max_cycles
        # place the employed bees on the solutions in the population
        # calculate initial best sol
        # print(self.population)
        while self.current_cycle < self.maximum_cycle_number:
            # each employed bee searches for an improvement
            for i in range(len(self.population)):
                # unpack
                cur_mapping, cur_alloc_dict, cur_cost, improvement_counter = self.population[i][:-1]

                for s in range(self.swaps_per_bee):
                    # Probabilistically produce modifications to try and improve the solution fitness}
                    new_mapping, new_alloc_dict = self.alloc_helper.random_swap(cur_mapping, cur_alloc_dict)
                    new_cost = self.alloc_helper.total_comm_cost(new_alloc_dict)

                    if new_cost < cur_cost:
                        # Greedily save the better solution
                        self.population[i] = [new_mapping, new_alloc_dict, new_cost, 0, 0]

                        # store the best solution so far
                        if new_cost < self.best_cost:
                            self.best_mapping = new_mapping
                            self.best_allocation = new_alloc_dict
                            self.best_cost = new_cost

                            print("New best cost: ", self.best_cost)
                            self.updated = True

                            # reset current cycle
                            self.current_cycle = 0

            # employed bees return to hive and do their dance
            # equivalent to building the roulette wheel selection policy
            self.build_roulette_wheel()

            # onlooker bees move to solutions based on shared fitnesses of existing solutions
            for bee in range(500*self.population_size):
                index_onlooker_move_to = self.roulette_wheel()
                self.population[index_onlooker_move_to][4] += 1

            # iterate over all solutions
            for i in range(len(self.population)):
                # unpack
                cur_mapping, cur_alloc_dict, cur_cost, improvement_counter, num_onlookers = self.population[i]

                improvement_counter += 1
                # for each onlooker at this solution
                for bee in range(num_onlookers):
                    for s in range(self.swaps_per_bee):
                        # Probabilistically produce modifications to try and improve the solution fitness}
                        new_mapping, new_alloc_dict = self.alloc_helper.random_swap(cur_mapping, cur_alloc_dict)
                        new_cost = self.alloc_helper.total_comm_cost(new_alloc_dict)

                        # increment solution improvement counter
                        if new_cost < cur_cost:
                            # Greedily save the better solution
                            cur_mapping, cur_alloc_dict, cur_cost = new_mapping, new_alloc_dict, new_cost

                            self.population[i] = [cur_mapping, cur_alloc_dict, cur_cost, 0, num_onlookers]

                            # store the best solution so far
                            if new_cost < self.best_cost:
                                self.best_mapping = new_mapping
                                self.best_allocation = new_alloc_dict
                                self.best_cost = new_cost

                                print("New best cost: ", self.best_cost)
                                self.updated = True

                                # reset current cycle
                                self.current_cycle = 0

                        elif improvement_counter > self.solution_improvement_limit:
                            # abandon food source
                            # employed bee moves from old to new food source
                            self.population[i] = [self.alloc_helper.generate_random_mapping(), 0, 0]

            # onlooker bees return to hive
            for i in range(self.population_size):
                # reset currently allocated onlookers at solution
                self.population[i][-1] = 0

            # increment cycle count
            if not self.updated:
                self.current_cycle += 1

        print("\nBest mapping has cost: ", self.best_cost)
        print(self.best_mapping)
        return self.best_mapping, self.best_cost


if __name__ == "__main__":
    try:
        filename = sys.argv[1]
        pop_size = sys.argv[2]
        cut_off = sys.argv[3]
        debug = False
        abc = ABCMapper(filename, pop_size, cut_off, debug)
        abc.map()
    except IndexError as msg:
        print(msg)
        exit()
