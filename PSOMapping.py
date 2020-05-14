import functools
import math
import sys
import random
import pickle
import Allocation
import CastNetMapping
import WCTG_Initialiser


class Particle:
    def __init__(self, alloc_helper, num_nodes, num_tiles, mesh_dims):
        self.num_nodes = num_nodes
        self.num_tiles = num_tiles
        self.mesh_dims = mesh_dims
        self.alloc_helper = alloc_helper

        self.local_best_cost = None
        self.local_best_mapping = None
        self.local_best_allocation = None

        self.current_mapping = None
        self.current_allocation = None
        self.current_cost = None
        self.initial_solution(self.alloc_helper)

    def initialise_from_existing_map(self, map):
        self.current_mapping, self.current_allocation, self.current_cost = map[0], map[1], map[2]
        self.local_best_mapping, self.local_best_allocation, self.local_best_cost = \
            self.current_mapping, self.current_allocation, self.current_cost

    def initial_solution(self, alloc_helper):
        self.current_mapping, self.current_allocation, self.current_cost = alloc_helper.generate_random_mapping()
        self.local_best_mapping, self.local_best_allocation, self.local_best_cost = \
            self.current_mapping, self.current_allocation, self.current_cost

    def compute_swap_sequence(self, source_seq, dest_seq):
        swap_seq = [None for _ in source_seq]
        for i in range(len(source_seq)):
            swap_seq[i] = dest_seq.tolist().index(int(source_seq[i]))
            if source_seq[i] == 0:
                dest_seq[dest_seq.tolist().index(int(source_seq[i]))] = self.num_tiles + 1
                source_seq[i] = self.num_tiles + 1
        return swap_seq

    def identify_local_swap_sequence(self):
        return self.compute_swap_sequence(self.current_mapping.flatten(), self.local_best_mapping.flatten())

    def identify_global_swap_sequence(self, global_best_mapping):
        return self.compute_swap_sequence(self.current_mapping.flatten(), global_best_mapping.flatten())

    def apply_swap_sequence(self, prob, swap_seq):
        current_swaps = self.current_mapping.flatten()
        did_swap = False
        for i in range(len(swap_seq)):
            rand_val = random.uniform(0, 1)
            if rand_val < prob:
                # swap current_swaps[i] with current_swaps[swap_seq[i]]
                temp = current_swaps[i]
                current_swaps[i] = current_swaps[swap_seq[i]]
                current_swaps[swap_seq[i]] = temp
                did_swap = True

            # reshape into a mapping
        self.current_mapping = current_swaps.reshape(self.mesh_dims)
        # print(current_swaps)
        self.current_allocation = self.alloc_helper.get_allocation_dict_from_mapping(self.current_mapping)
        return did_swap


class PSOMapper:
    def __init__(self, file, num_particles, cutoff, debug):
        print("***PSO Mapping***")
        print("Reading in file...")
        self.num_nodes, self.node_weights, self.comm_matrix = WCTG_Initialiser.initialise_wctg_from_dataset_random(file)
        print("Initialised WCTG.")

        # allocation helper
        self.side_len = math.ceil(math.sqrt(self.num_nodes))
        self.mesh_dims = [self.side_len, self.side_len]
        self.num_tiles = functools.reduce(lambda x, y: (x * y), self.mesh_dims, 1)
        self.alloc_helper = Allocation.Allocation(self.num_nodes, self.comm_matrix, self.mesh_dims, self.node_weights)

        ca = CastNetMapping.CastNetMapper(filename, debug)
        self.ca_maps = ca.map()
        # with open('../Results/CastNet256.pkl', 'rb') as f:
        #     self.ca_maps = pickle.load(f)

        # print("CastNet: ", self.ca_maps)
        # self.ca_maps = math.floor(self.num_tiles / len(self.ca_maps)) * self.ca_maps
        # l = len(self.ca_maps)

        self.prob_local_swap = 0.04         # FROM PAPER
        self.prob_global_swap = 0.02        # FROM PAPER

        self.global_best_cost = math.inf
        self.global_best_mapping = None
        self.global_best_allocation = None
        self.seen_mappings = []
        self.particles = self.generate_particles(num_particles)
        print("Num particles:", len(self.particles))

        self.current_cycle = 0
        self.cutoff = cutoff

        # debug
        self.debug = debug
        print("Initialised.")

    def generate_particles(self, num_particles):
        particles = []
        for _ in range(num_particles):
            p = Particle(self.alloc_helper, self.num_nodes, self.num_tiles, self.mesh_dims)
            if p.current_cost < self.global_best_cost:
                self.global_best_cost = p.current_cost
                self.global_best_mapping = p.current_mapping
                self.global_best_allocation = p.current_allocation
            particles.append(p)
        for ca_map in self.ca_maps:
            p = Particle(self.alloc_helper, self.num_nodes, self.num_tiles, self.mesh_dims)
            p.initialise_from_existing_map(ca_map)
            if p.current_cost < self.global_best_cost:
                self.global_best_cost = p.current_cost
                self.global_best_mapping = p.current_mapping
                self.global_best_allocation = p.current_allocation
            particles.append(p)
        return particles

    def map(self):
        print("Initial best cost:", self.global_best_cost)
        while self.current_cycle < self.cutoff:
            for p in self.particles:
                # identify local swap sequence
                lss = p.identify_local_swap_sequence()
                # identify global swap sequences
                gss = p.identify_global_swap_sequence(self.global_best_mapping)

                # apply local swap sequence with p = s2
                did_local_swap = p.apply_swap_sequence(self.prob_local_swap, lss)

                # apply global swap sequence with p = s3
                did_global_swap = p.apply_swap_sequence(self.prob_global_swap, gss)

                # evaluate fitness of particle
                p.current_cost = self.alloc_helper.total_comm_cost(p.current_allocation)

                # if better than local cost, update local cost
                if p.current_cost < p.local_best_cost:
                    p.local_best_cost = p.current_cost
                    p.local_best_mapping = p.current_mapping
                    p.local_best_allocation = p.current_allocation

                # if better than global cost, update global cost
                if p.current_cost < self.global_best_cost:
                    self.global_best_cost = p.current_cost
                    self.global_best_mapping = p.current_mapping
                    self.global_best_allocation = p.current_allocation
                    print("New best cost: ", self.global_best_cost)

                    # reset cycle, as have updated best cost
                    self.current_cycle = 0
                pass
            self.current_cycle += 1

        print("\nBest mapping has cost: ", self.global_best_cost)
        print(self.global_best_mapping)


if __name__ == "__main__":
    try:
        filename = sys.argv[1]
        num_particles = int(sys.argv[2])
        cut_off = int(sys.argv[3])
        debug = False
        pso = PSOMapper(filename, num_particles, cut_off, debug)
        pso.map()
    except IndexError as msg:
        print(msg)
        exit()
