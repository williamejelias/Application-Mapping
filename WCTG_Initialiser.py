import numpy as np


def initialise_wctg_from_dataset_random(filename):
    # read all lines from file
    with open(filename) as f:
        content = f.readlines()

    # extract number of nodes and respective weights
    num_nodes = int(content[0].strip())
    node_weights = [float(x) for x in content[1].strip().split()]
    print(num_nodes, "nodes")
    print("Node weights: ", node_weights)

    # empty 2D square matrix with side length number of nodes
    adj_matrix = np.empty(shape=(num_nodes, num_nodes))
    adj_matrix.fill(0)

    # populate the square matrix with the edge weights
    i = 2
    column_index = 1
    while i < len(content):
        row = i - 2
        values = [float(x) for x in content[i].strip().split()]
        # print("Row: ", row, " - ", values)
        for column in range(column_index, num_nodes):
            adj_matrix[row][column] = values[column]
            adj_matrix[column][row] = values[column]
        i += 1
        column_index += 1
    print("Adj Matrix: ", adj_matrix)

    e = 0
    for x in np.nditer(adj_matrix):
        if x > 0:
            e += 1
    print("Edges: ", e/2)

    return num_nodes, node_weights, adj_matrix


# if __name__ == "__main__":
#     import sys
#     filename = sys.argv[1]
#     initialise_wctg_from_dataset_random(filename)

# 1.17s 504366
# 1634s 478270
# 1660s 481632
