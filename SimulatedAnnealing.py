import math
import random
import sys
from matplotlib import pyplot as plt
import functools
import Allocation
import WCTG_Initialiser


class SimulatedAnnealingMapper:
    def __init__(self, file, debug=False):
        print("***Simulated Annealing Mapper***")
        print("Reading in file...")
        self.num_nodes, self.node_weights, self.comm_matrix = WCTG_Initialiser.initialise_wctg_from_dataset_random(file)
        print("Initialised WCTG.")
        self.debug = debug
        self.data = []

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')

        ax.set_ylabel("Communication Costs", fontsize=14)
        ax.set_xlabel("Iterations", fontsize=14)
        label = "Simulated Annealing Algorithm MPEG4"
        plt.title(label)
        plt.plot(self.data, color='green')
        plt.show()
        plt.savefig("../Results/SA-MPEG4.png")

    def map(self):
        print("\nAnnealing...")
        side_len = math.ceil(math.sqrt(self.num_nodes))
        mesh_dims = [side_len, side_len]
        num_tiles = functools.reduce(lambda x, y: (x * y), mesh_dims, 1)
        alloc_helper = Allocation.Allocation(self.num_nodes, self.comm_matrix, mesh_dims, self.node_weights)

        # create 2D square basic mesh and randomly allocate
        initial_mapping, initial_allocation_dict, initial_cost = alloc_helper.generate_random_mapping()

        # generate 100 initial maps and start with the best of those
        for i in range(100):
            mapping, allocation_dict, cost = alloc_helper.generate_random_mapping()
            if cost < initial_cost:
                initial_cost = cost
                initial_mapping, initial_allocation_dict = mapping, allocation_dict

        best_mapping = initial_mapping
        best_allocation_dict = initial_allocation_dict
        best_cost = initial_cost

        #
        cost = initial_cost
        allocation_dict = initial_allocation_dict
        mapping = initial_mapping

        # define starting temperature
        temp = math.ceil(10 * math.log(num_tiles))
        print("Initial Temperature:", temp)

        # pseudo-code of SA method from main paper
        for i in range(num_tiles ** 2):
            r = 0
            while r < 1000:
                # new mapping random swap on existing mapping
                new_mapping, new_alloc_dict = alloc_helper.random_swap(mapping, allocation_dict)

                # calculate the cost of new mapping
                new_cost = alloc_helper.total_comm_cost(new_alloc_dict)

                # calculate c_delta = c - c'
                c_delta = new_cost - cost

                # generate a random variable in range 0-1
                alpha = random.uniform(0, 1)
                if c_delta <= 0 or alpha <= math.e ** (-c_delta / temp):
                    mapping = new_mapping
                    allocation_dict = new_alloc_dict
                    cost = new_cost
                    r = 0
                else:
                    r += 1

                if r == 0 and new_cost < best_cost:
                    print("new best cost: ", new_cost, 'temp: ', temp)

                    best_mapping = new_mapping
                    best_allocation_dict = new_alloc_dict
                    best_cost = new_cost

                self.data.append(cost)
            temp *= 0.99

        print("\nBest mapping has cost: ", best_cost, temp)
        print(best_mapping)


if __name__ == "__main__":
    try:
        filename = sys.argv[1]
        debug = False
        sa = SimulatedAnnealingMapper(filename, debug)
        sa.map()
        # sa.plot()
        # simulated_annealing_mapping(filename)
    except IndexError as msg:
        print(msg)
        exit()
