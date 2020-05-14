import random
import numpy as np
import itertools


class Allocation:
    def __init__(self, num_nodes, comm_matrix, mesh_dims, node_weights=None):
        self.num_nodes = num_nodes
        self.mesh_dims = mesh_dims
        self.comm_matrix = comm_matrix
        self.possible_locs = self.generate_possible_coords()
        self.node_weights = node_weights

    # return the manhattan distance between coordinates
    @staticmethod
    def manhattan(c1, c2):
        m = 0
        for dim in range(len(c1)):
            val = abs(c2[dim] - c1[dim])
            m += val
        return m

    def generate_possible_coords(self):
        coords = [[i] for i in range(self.mesh_dims[0])]
        for dim in self.mesh_dims[1:]:
            new_coords = []
            for i in range(dim):
                for c in coords:
                    new_c = c.copy()
                    new_c.append(i)
                    new_coords.append(tuple(new_c))
            coords = new_coords
        return coords

    # generates a point tuple within the list of given dimensions
    def generate_random_coordinate(self):
        index = random.randint(0, len(self.possible_locs) - 1)
        return self.possible_locs[index]

    @staticmethod
    def choose_random_coordinate(poss_coords):
        index = random.randint(0, len(poss_coords) - 1)
        return poss_coords[index]

    # find the total communication cost of a Mapping
    def total_comm_cost(self, allocation_dict):
        total_cost = 0
        for n1_index in range(self.num_nodes - 1):
            for n2_index in range(n1_index + 1, self.num_nodes):
                edge_weight = self.comm_matrix[n1_index][n2_index]
                if edge_weight == 0:
                    pass
                else:
                    n1_id, n2_id = n1_index + 1, n2_index + 1
                    n1_coords, n2_coords = allocation_dict[n1_id], allocation_dict[n2_id]
                    m = self.manhattan(n1_coords, n2_coords)
                    pair_cost = m * edge_weight
                    total_cost += pair_cost
        return total_cost

    # find the communication of a partial mapping
    def partial_comm_cost(self, allocation_dict):
        total_cost = 0
        mapped_nodes = allocation_dict.keys()
        node_pairs = list(itertools.combinations(mapped_nodes, 2))
        for node_pair in node_pairs:
            n1_id, n2_id = node_pair[0], node_pair[1]
            n1_index, n2_index = n1_id - 1, n2_id - 1
            edge_weight = self.comm_matrix[n1_index][n2_index]
            if edge_weight == 0:
                pass
            else:
                n1_coords, n2_coords = allocation_dict[n1_id], allocation_dict[n2_id]
                m = self.manhattan(n1_coords, n2_coords)
                pair_cost = m * edge_weight
                total_cost += pair_cost
        return total_cost

    def random_swap(self, mapping, alloc_dict):
        # generate two random coordinates and swap the nodes at those tiles
        # could either move a node into an empty tile or swap two nodes
        t1_loc = self.generate_random_coordinate()
        t2_loc = self.generate_random_coordinate()
        while t2_loc == t1_loc:
            t2_loc = self.generate_random_coordinate()

        node1 = mapping[t1_loc]
        node2 = mapping[t2_loc]
        if node1 == 0 and node2 == 0:
            pass
        elif node1 == 0:
            # tile 1 has no node on, so move node 2 to its location
            alloc_dict[node2] = t1_loc
            mapping[t1_loc] = node2
            mapping[t2_loc] = 0
        elif node2 == 0:
            # tile 2 has no node on, so move node 1 to its location
            alloc_dict[node1] = t2_loc
            mapping[t2_loc] = node1
            mapping[t1_loc] = 0
        else:
            # swap both nodes
            alloc_dict[node2] = t1_loc
            alloc_dict[node1] = t2_loc
            mapping[t2_loc] = node1
            mapping[t1_loc] = node2
        return mapping, alloc_dict

    # mesh dims is a list of dims i.e. (2, 3), or (3, 3, 3)
    def generate_random_mapping(self):
        mesh = np.zeros(shape=self.mesh_dims)

        # track already allocated locations
        poss_coords = self.generate_possible_coords()
        allocation_dict = {}

        # all nodes
        for node in range(self.num_nodes):
            # node ids must start at 1 in order tha 0 can represent an empty space
            node_id = node + 1

            # generate random coordinate to enter node
            location = self.choose_random_coordinate(poss_coords)
            poss_coords.remove(location)
            mesh[location] = node_id
            allocation_dict[node_id] = location
        return mesh, allocation_dict, self.total_comm_cost(allocation_dict)

    def generate_empty_mesh(self):
        mesh = np.zeros(shape=self.mesh_dims)
        return mesh

    def get_allocation_dict_from_mapping(self, mapping):
        alloc_dict = {}
        for loc in self.possible_locs:
            node = int(mapping[loc])
            if node != 0:
                alloc_dict[node] = loc
        return alloc_dict

