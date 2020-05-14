import functools
import itertools
import math
import sys

from ortools.linear_solver import pywraplp

import Allocation
import WCTG_Initialiser


class ILPMapper:
    def __init__(self, file, debug):
        print("***ILP Mapping***")
        print("Reading in file...")
        self.num_nodes, self.node_weights, self.comm_matrix = WCTG_Initialiser.initialise_wctg_from_dataset_random(file)
        print("Initialised WCTG.")

        # variables
        self.alpha_vars = None
        self.alpha_locs = None
        self.x_pair_dist_vars = None
        self.y_pair_dist_vars = None
        self.var_pairs = None

        # objective variables
        self.X_Cost = None
        self.Y_Cost = None

        self.solver = pywraplp.Solver('ApplicationMapping',
                                      pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        self.objective = self.solver.Objective()

        # debug
        self.debug = debug

    def build_variables(self, X_Dim, Y_Dim):
        self.alpha_vars = {}
        self.alpha_locs = {}
        # alpha variables
        # one dict for each node to map
        for t in range(1, self.num_nodes + 1):
            self.alpha_vars[t] = {}
            # for each node, have a variable for each mesh location
            # variable is 0 or 1 depending on whether node t is allocated to i, j on mesh
            for i in range(X_Dim):
                for j in range(Y_Dim):
                    name = str(t) + "-" + str(i) + "-" + str(j)
                    var = self.solver.IntVar(0, 1, name)
                    self.alpha_vars[t][(i, j)] = var

        for i in range(X_Dim):
            for j in range(Y_Dim):
                self.alpha_locs[(i, j)] = {}
                # for each node, have a variable for each mesh location
                # variable is 0 or 1 depending on whether node t is allocated to i, j on mesh
                for t in range(1, self.num_nodes + 1):
                    self.alpha_locs[(i, j)][t] = self.alpha_vars[t][(i, j)]

        # all ordered pairs no repeats (i, j)
        self.var_pairs = list(itertools.combinations([i + 1 for i in range(self.num_nodes)], 2))
        if self.debug:
            print("Task-Task pairs: ", self.var_pairs)

        # X Dist vars
        # for every pair of tasks, var for each possible X Dist between those tasks in an allocation
        self.x_pair_dist_vars = {}
        for pair in self.var_pairs:
            # for each pair of nodes, X_i,j,a = 1 if X distance between i,j is a
            self.x_pair_dist_vars[pair] = {}
            for x_val in range(X_Dim):
                name = "X-" + str(pair[0]) + "-" + str(pair[1]) + "-" + str(x_val)
                self.x_pair_dist_vars[pair][x_val] = self.solver.IntVar(0, 1, name)

        # Y Dist vars
        # for every pair of tasks, var for each possible Y Dist between those tasks in an allocation
        self.y_pair_dist_vars = {}
        for pair in self.var_pairs:
            # for each pair of nodes, Y_i,j,a = 1 if Y distance between i,j is a
            self.y_pair_dist_vars[pair] = {}
            for y_val in range(Y_Dim):
                name = "Y-" + str(pair[0]) + "-" + str(pair[1]) + "-" + str(y_val)
                self.y_pair_dist_vars[pair][y_val] = self.solver.IntVar(0, 1, name)

        # define objective vars
        self.X_Cost = self.solver.NumVar(0, self.solver.infinity(), "X_Cost")
        self.Y_Cost = self.solver.NumVar(0, self.solver.infinity(), "Y_Cost")

    def build_constraints(self, X_Dim, Y_Dim):
        # each task is mapped constraint to only one tile
        print("\nEach task is mapped to only one tile.")
        for t in range(1, self.num_nodes + 1):
            all_nodes_mapped_constraint = self.solver.Constraint(1, 1)
            if self.debug:
                print("\nSum of all = 1")
            for location, variable in self.alpha_vars[t].items():
                if self.debug:
                    print(t, location, variable)
                all_nodes_mapped_constraint.SetCoefficient(variable, 1)

        # all locations may have up to 1 task mapped
        print("\nEach tile may have 0 or 1 task mapped.")
        for i in range(X_Dim):
            for j in range(Y_Dim):
                node_has_0_or_1_task_constraint = self.solver.Constraint(0, 1)
                # node_has_0_or_1_task_constraint = self.solver.Constraint(-self.solver.infinity(), 1)
                if self.debug:
                    print("\nSum of all = 0 or 1")
                for variable, location in self.alpha_locs[(i, j)].items():
                    if self.debug:
                        print("Tile: ", (i, j), " task: ", variable, " variable: ", location)
                    node_has_0_or_1_task_constraint.SetCoefficient(location, 1)

        # X_Dist_i,j,a - alpha_i,xi,yi - alpha_j,xj,yj >= -1        forall i,j in E
        # Y_Dist_i,j,a - alpha_i,xi,yi - alpha_j,xj,yj >= -1        forall i,j in E
        locs = [(i, j) for i in range(X_Dim) for j in range(Y_Dim)]
        loc_pairs = list(itertools.permutations(locs, 2))

        if self.debug:
            print("Location pairs: ", loc_pairs)
            print("For each task pair:")
        for pair in self.var_pairs:
            i, j = pair[0], pair[1]

            # X_Dist constraints:
            for x_dist_var in self.x_pair_dist_vars[pair]:
                if self.debug:
                    print("Pair:", pair, "X dist=", x_dist_var, self.x_pair_dist_vars[pair][x_dist_var])
                for loc_pair in loc_pairs:
                    i_alloc, j_alloc = loc_pair[0], loc_pair[1]
                    if abs(j_alloc[0] - i_alloc[0]) == x_dist_var:
                        if self.debug:
                            print("Constraint: ", self.x_pair_dist_vars[pair][x_dist_var], " - ", self.alpha_vars[i][i_alloc], " - ", self.alpha_vars[j][j_alloc], " >= -1")
                        x_dist_between_tasks_constraint = self.solver.Constraint(-1, self.solver.infinity())
                        x_dist_between_tasks_constraint.SetCoefficient(self.x_pair_dist_vars[pair][x_dist_var], 1)
                        x_dist_between_tasks_constraint.SetCoefficient(self.alpha_vars[i][i_alloc], -1)
                        x_dist_between_tasks_constraint.SetCoefficient(self.alpha_vars[j][j_alloc], -1)

            # Y_Dist constraints:
            for y_dist_var in self.y_pair_dist_vars[pair]:
                if self.debug:
                    print("Pair:", pair, "Y dist=", y_dist_var, self.y_pair_dist_vars[pair][y_dist_var])
                for loc_pair in loc_pairs:
                    i_alloc, j_alloc = loc_pair[0], loc_pair[1]
                    if abs(j_alloc[1] - i_alloc[1]) == y_dist_var:
                        if self.debug:
                            print("Constraint: ", self.y_pair_dist_vars[pair][y_dist_var], " - ", self.alpha_vars[i][i_alloc], " - ", self.alpha_vars[j][j_alloc], " >= -1")
                        y_dist_between_tasks_constraint = self.solver.Constraint(-1, self.solver.infinity())
                        y_dist_between_tasks_constraint.SetCoefficient(self.y_pair_dist_vars[pair][y_dist_var], 1)
                        y_dist_between_tasks_constraint.SetCoefficient(self.alpha_vars[i][i_alloc], -1)
                        y_dist_between_tasks_constraint.SetCoefficient(self.alpha_vars[j][j_alloc], -1)

        # X_Cost Constraints...
        print("X_Cost - forall pairs tasks i,j; for a=1 to X_Dim; w_i,j * a * X_Dist_i,j,a == 0")
        x_cost_constraint = self.solver.Constraint(0, 0)
        x_cost_constraint.SetCoefficient(self.X_Cost, 1)

        # Y_Cost Constraints...
        print("Y_Cost - forall pairs tasks i,j; for a=1 to Y_Dim; w_i,j * a * Y_Dist_i,j,a == 0")
        y_cost_constraint = self.solver.Constraint(0, 0)
        y_cost_constraint.SetCoefficient(self.Y_Cost, 1)

        # build X_Cost and Y_Cost constraints
        # for every pair of tasks
        for pair in self.var_pairs:
            # X_Dist X_Cost constraints:
            for x_dist, x_dist_var in self.x_pair_dist_vars[pair].items():
                i, j = pair[0], pair[1]
                edge_weight = self.comm_matrix[i-1][j-1]
                if edge_weight != 0:
                    x_cost_constraint.SetCoefficient(x_dist_var, -edge_weight*x_dist)
                    if self.debug:
                        print("Tasks:", i, j, "have x_dist:", x_dist, "vname:", x_dist_var, "with edge weight:", edge_weight)
                        print("X_Cost += ", x_dist_var, " * ", edge_weight, " * ", x_dist)

            # Y_Dist Y_Cost constraints:
            for y_dist, y_dist_var in self.y_pair_dist_vars[pair].items():
                i, j = pair[0], pair[1]
                edge_weight = self.comm_matrix[i - 1][j - 1]
                if edge_weight != 0:
                    y_cost_constraint.SetCoefficient(y_dist_var, -edge_weight * y_dist)
                    if self.debug:
                        print("Tasks:", i, j, "have y_dist:", y_dist, "vname:", y_dist_var, "with edge weight:", edge_weight)
                        print("Y_Cost += ", y_dist_var, " * ", edge_weight, " * ", y_dist)

    def build_objective(self):
        # minimize COMM = X_Cost + Y_Cost
        self.objective.SetCoefficient(self.X_Cost, 1)
        self.objective.SetCoefficient(self.Y_Cost, 1)
        self.objective.SetMinimization()

    def map(self):
        # create 2D square basic mesh and randomly allocate
        print("\nFormulating problem...")
        side_len = math.ceil(math.sqrt(self.num_nodes))
        mesh_dims = [side_len, side_len]
        X_Dim = side_len
        Y_Dim = side_len
        num_tiles = functools.reduce(lambda x, y: (x * y), mesh_dims, 1)

        alloc_helper = Allocation.Allocation(self.num_nodes, self.comm_matrix, mesh_dims, self.node_weights)

        print("Building variables...")
        self.build_variables(X_Dim, Y_Dim)
        print("Built variables.")

        print("Building constraints...")
        self.build_constraints(X_Dim, Y_Dim)
        print("Built constraints.")

        print("Building objective...")
        self.build_objective()
        print("Built objective.")

        print("Solving...")
        self.solve(alloc_helper)

    def solve(self, alloc_helper):
        # Solve!
        print()
        status = self.solver.Solve()

        if self.debug:
            for pair in self.var_pairs:
                # X_Dist X_Cost constraints:
                for x_dist, x_dist_var in self.x_pair_dist_vars[pair].items():
                    i, j = pair[0], pair[1]
                    edge_weight = self.comm_matrix[i-1][j-1]
                    if edge_weight != 0:
                        print("X_Cost += ", x_dist_var, x_dist_var.solution_value(), " * ", edge_weight, " * ", x_dist)

            for pair in self.var_pairs:
                # Y_Dist Y_Cost constraints:
                for y_dist, y_dist_var in self.y_pair_dist_vars[pair].items():
                    i, j = pair[0], pair[1]
                    edge_weight = self.comm_matrix[i-1][j-1]
                    if edge_weight != 0:
                        print("Y_Cost += ", y_dist_var, y_dist_var.solution_value(), " * ", edge_weight, " * ", y_dist)

        print("Status: ", status)
        assert status == pywraplp.Solver.OPTIMAL
        print('Objective value =', self.solver.Objective().Value())
        print("Num Variables: ", self.solver.NumVariables())
        print("X Cost: ", self.X_Cost.solution_value())
        print("Y Cost: ", self.Y_Cost.solution_value())

        optimal_mapping = alloc_helper.generate_empty_mesh()
        optimal_alloc_dict = {}
        for t in range(1, self.num_nodes + 1):
            # for each node, have a variable for each mesh location
            # variable is 0 or 1 depending on whether node t is allocated to i, j on mesh
            for i in range(4):
                for j in range(4):
                    if self.alpha_vars[t][(i, j)].solution_value() == 1:
                        optimal_mapping[(i, j)] = t
                        optimal_alloc_dict[t] = (i, j)
                        print(t, " -> ", (i, j))

        print(optimal_mapping)
        print(optimal_alloc_dict)
        print(alloc_helper.total_comm_cost(optimal_alloc_dict))

        result_string = "*** Application Mapping ***\n"
        if status == pywraplp.Solver.OPTIMAL:
            result_string += "An optimal mapping was found for this WCTG and Mesh"
        else:  # No optimal solution was found.
            if status == pywraplp.Solver.FEASIBLE:
                string = "A potentially suboptimal solution was found.\n"
                result_string += string
            else:
                string = "The solver could not solve the problem.\n"
                result_string += string
        print(result_string)
        return result_string


if __name__ == "__main__":
    try:
        filename = sys.argv[1]
        debug = False
        ilp = ILPMapper(filename, debug)
        ilp.map()
    except IndexError as msg:
        print(msg)
        exit()
