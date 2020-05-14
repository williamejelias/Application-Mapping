import functools
import math
import sys
import numpy as np
import Allocation
import WCTG_Initialiser
import pickle


class CastNetMapper:
    def __init__(self, file, debug=False):
        print("***CastNet Mapping***")
        print("Reading in file...")
        self.num_nodes, self.node_weights, self.comm_matrix = WCTG_Initialiser.initialise_wctg_from_dataset_random(file)
        self.debug = debug
        print("Initialised WCTG.\n")

        self.connections_dict = {}
        self.priority_list = self.build_priority_list()
        self.tasks = [i for i in range(1, self.num_nodes + 1)]

    def build_priority_list(self):
        if self.debug:
            print("Building task priority list...")
        tasks_priorities = {i: {'total': 0, 'links': 0} for i in range(1, self.num_nodes + 1)}
        self.connections_dict = {i: [] for i in range(1, self.num_nodes + 1)}
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                score = self.comm_matrix[i, j]
                if score != 0:
                    tasks_priorities[i + 1]['total'] += score
                    tasks_priorities[j + 1]['total'] += score
                    tasks_priorities[i + 1]['links'] += 1
                    tasks_priorities[j + 1]['links'] += 1
                    self.connections_dict[i + 1].append(j + 1)
                    self.connections_dict[j + 1].append(i + 1)

        priority_tasks = sorted(tasks_priorities, key=lambda k: (
            tasks_priorities[k]['total'], tasks_priorities[k]['total'] / tasks_priorities[k]['links']), reverse=True)
        if self.debug:
            print(tasks_priorities)
            print("Tasks in order of priority: ", priority_tasks)
            print("Connections dict: ", self.connections_dict)
        return priority_tasks

    def highest_priority(self, unmapped_tasks):
        highest_priority = [i for i in self.priority_list if i in unmapped_tasks][0]
        if self.debug:
            print("Highest priority task: ", highest_priority)
        return highest_priority

    # assumes a square symmetrical mesh
    def get_candidates(self, side_len):
        # returns a list of candidate tiles in the top left symmetrical segment group of the square mesh
        if self.debug:
            print("Generating candidate tiles...")

        candidates = []
        for i in range(math.ceil(side_len / 2)):
            for j in range(i + 1):
                candidates.append((i, j))

        if self.debug:
            print("Candidate tiles:", candidates)
        return candidates

    def select_task(self, mapped_tasks):
        # get all connected tasks
        connected_tasks = []
        for tm in mapped_tasks:
            connected_tasks += self.connections_dict[tm]
        # remove already mapped tasks
        connected_tasks = [tc for tc in connected_tasks if tc not in mapped_tasks]

        if self.debug:
            print("Connected tasks: ", connected_tasks)

        # iterate over connected tasks to choose task with highest communication
        max_comm_cost = 0
        maxes = []
        task_values = {}
        for ct in connected_tasks:
            sum = 0
            for tm in mapped_tasks:
                sum += self.comm_matrix[ct - 1, tm - 1]
            if sum > max_comm_cost:
                max_comm_cost = sum
                maxes = [ct]
            elif sum == max_comm_cost:
                maxes.append(ct)
            task_values[ct] = sum
        unmapped_tasks_sorted_by_value = sorted(task_values, key=lambda k: (task_values[k]), reverse=True)

        if self.debug:
            print("Connected tasks comm values: ", task_values)
            print("Connected tasks sorted by comm value: ", unmapped_tasks_sorted_by_value)

        # if more than one task has same comm cost, choose the one with the highest priority
        if len(maxes) > 1:
            index = self.num_nodes
            highest_priority = maxes[0]
            for m in maxes:
                if self.priority_list.index(m) < index:
                    index = self.priority_list.index(m)
                    highest_priority = m
            return highest_priority
        return unmapped_tasks_sorted_by_value[0]

    def surrounding_locs(self, loc):
        dims = len(loc)
        poss_moves = []
        diffs = [-1, 1]
        for dim in range(dims):
            for diff in diffs:
                loc_copy = list(loc)
                loc_copy[dim] += diff
                poss_moves.append(tuple(loc_copy))
        return poss_moves

    def select_core(self, alloc_helper, current_mapping, current_alloc_dict, target_task, unmapped_tasks,
                    unmapped_cores):
        min_cost = math.inf
        min_cost_locs = []
        for loc in unmapped_cores:
            test_mapping = np.copy(current_mapping)
            test_alloc_dict = dict(current_alloc_dict)

            # map the target task to the location and check the comm cost
            test_mapping[loc] = target_task
            test_alloc_dict[target_task] = loc
            test_cost = alloc_helper.partial_comm_cost(test_alloc_dict)
            if test_cost < min_cost:
                min_cost = test_cost
                min_cost_locs = [loc]
            elif test_cost == min_cost:
                min_cost_locs.append(loc)

        if self.debug:
            print("Min cost: ", min_cost)
            print("Possible mapping locations: ", min_cost_locs)

        current_task_num_edges = len([i for i in self.connections_dict[target_task] if i in unmapped_tasks])
        if self.debug:
            print("Num edges current task: ", current_task_num_edges)

        if len(min_cost_locs) == 1:
            return min_cost_locs[0]
        else:
            # calculate the number of free locations that are a one-hop distance away
            min_abs = math.inf
            min_abs_candidates = []
            max_abs = 0
            max_abs_candidates = []
            for loc in min_cost_locs:
                surrounding_locs = self.surrounding_locs(loc)
                num_free_onehop_locations_around_core = len([c for c in surrounding_locs if c in unmapped_cores])
                cur_abs_diff = abs(current_task_num_edges - num_free_onehop_locations_around_core)
                if current_task_num_edges - num_free_onehop_locations_around_core < 0:
                    if cur_abs_diff > max_abs:
                        max_abs = cur_abs_diff
                        max_abs_candidates = [loc]
                    elif cur_abs_diff == max_abs:
                        max_abs_candidates.append(loc)
                else:
                    if cur_abs_diff < min_abs:
                        min_abs = cur_abs_diff
                        min_abs_candidates = [loc]
                    elif cur_abs_diff == min_abs:
                        min_abs_candidates.append(loc)
            candidate_cores = max_abs_candidates + min_abs_candidates
            if self.debug:
                print("Candidate cores: ", candidate_cores)
            return candidate_cores[0]

    def map(self):
        print("\nExecuting Heuristic CastNet Algorithm...")
        side_len = math.ceil(math.sqrt(self.num_nodes))
        mesh_dims = [side_len, side_len]
        num_tiles = functools.reduce(lambda x, y: (x * y), mesh_dims, 1)

        alloc_helper = Allocation.Allocation(self.num_nodes, self.comm_matrix, mesh_dims, self.node_weights)

        print("Prioritising tasks...")
        priority_task = self.highest_priority(self.tasks)
        print("Generating candidate start locations...")
        candidate_locations = self.get_candidates(side_len)
        best_comm_cost = math.inf
        best_mapping = None
        best_alloc_dict = None

        comm_costs = []

        # CastNet algorithm
        for loc in candidate_locations:
            print("\nChecking alternative candidate location...")
            # make a blank mesh
            mapping = alloc_helper.generate_empty_mesh()
            mapped_tasks = []
            unmapped_tasks = [i for i in range(1, self.num_nodes + 1)]
            mapped_cores = []
            unmapped_cores = list(alloc_helper.possible_locs)

            # map the highest priority task to the candidate location
            mapping[loc] = priority_task
            alloc_dict = alloc_helper.get_allocation_dict_from_mapping(mapping)
            mapped_tasks.append(priority_task)
            unmapped_tasks.remove(priority_task)
            mapped_cores.append(loc)
            unmapped_cores.remove(loc)

            print("Mapped highest priority task.")
            if self.debug:
                print("\nAllocated highest priority task:", priority_task, "to location:", loc)
                print(mapping)
                print(alloc_dict)
                print("Mapped tasks:", mapped_tasks)
                print("Unmapped tasks:", unmapped_tasks)
                print("Mapped cores:", mapped_cores)
                print("Unmapped cores:", unmapped_cores)
            print("Mapping remaining cores...")

            while len(unmapped_tasks) > 0:
                if self.debug:
                    print("\nMapped tasks:", mapped_tasks)
                    print("Unmapped tasks:", unmapped_tasks)
                    print("Mapped cores:", mapped_cores)
                    print("Unmapped cores:", unmapped_cores)
                    print(mapping)

                # choose next task to allocate
                target_task = self.select_task(mapped_tasks)
                if self.debug:
                    print("Chosen task: ", target_task)

                # choose next core to allocate to
                target_core = self.select_core(alloc_helper, mapping, alloc_dict, target_task, unmapped_tasks,
                                               unmapped_cores)
                if self.debug:
                    print("Chosen core: ", target_core)

                # do the allocation
                mapping[target_core] = target_task
                alloc_dict[target_task] = target_core
                mapped_tasks.append(target_task)
                unmapped_tasks.remove(target_task)
                mapped_cores.append(target_core)
                unmapped_cores.remove(target_core)

            total_comm_cost = alloc_helper.total_comm_cost(alloc_dict)
            result = (mapping, alloc_dict, total_comm_cost)
            if total_comm_cost < best_comm_cost:
                best_comm_cost = total_comm_cost
                best_mapping = mapping
                best_alloc_dict = alloc_dict
            comm_costs.append(result)
            # if self.debug:
            print("Total cost of mapping: ", total_comm_cost)
            print(mapping)
        print("\nBest mapping has cost: ", best_comm_cost)
        print(best_mapping)
        return comm_costs

    def plot(self):
        pass


if __name__ == "__main__":
    try:
        filename = sys.argv[1]
        debug = False
        ca = CastNetMapper(filename, debug)
        ca_maps = ca.map()
    except IndexError as msg:
        print(msg)
        exit()
