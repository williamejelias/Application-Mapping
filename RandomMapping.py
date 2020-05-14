import math
import sys
import Allocation
import WCTG_Initialiser


class RandomMapper:
    def __init__(self, file, num_reps, debug=False):
        print("***Random Mapper***")
        print("Reading in file...")
        self.num_nodes, self.node_weights, self.comm_matrix = WCTG_Initialiser.initialise_wctg_from_dataset_random(file)
        print("Initialised WCTG.")
        self.num_reps = num_reps
        self.debug = debug

    def map(self):
        print("\n\nGenerating Random Maps")
        side_len = math.ceil(math.sqrt(self.num_nodes))
        mesh_dims = [side_len, side_len]

        allocation = Allocation.Allocation(self.num_nodes, self.comm_matrix, mesh_dims, self.node_weights)

        mapping, allocation_dict, cost = allocation.generate_random_mapping()

        best_mapping = mapping
        lowest_cost = cost
        for i in range(self.num_reps):
            mapping, allocation_dict, cost = allocation.generate_random_mapping()
            if cost < lowest_cost:
                lowest_cost = cost
                best_mapping = mapping
        print("\n\nBest mapping after ", self.num_reps, " runs has cost: ", lowest_cost)
        print(best_mapping)


if __name__ == "__main__":
    try:
        filename = sys.argv[1]
        r = RandomMapper(filename, 1000)
        r.map()
    except IndexError as msg:
        print(msg)
        exit()
