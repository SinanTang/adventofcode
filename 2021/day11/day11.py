import numpy as np


def get_neighbors(idx: tuple, m: np.array):
    neighbors = set()  # use tuples to represent nodes

    x, y = idx
    max_x, max_y = len(m[0]) - 1, len(m) - 1

    for a in range(max(x-1, 0), min(x+1, max_x) + 1):
        for b in range(max(y-1, 0), min(y+1, max_y) + 1):
            if not (x == a and y == b):
                neighbors.add((a, b))

    return neighbors


def build_graph(m: np.array):
    graph = dict()

    for idx, val in np.ndenumerate(m):
        neighbors = get_neighbors(idx, m)
        graph[idx] = neighbors

    return graph


def after_one_step(graph: dict, m: np.array):
    # graph stores indices (x,y) -> hashable
    # matrix stores values -> mutable
    flashed = set()
    stack = list(graph.keys())  # all nodes need to be visited at least once

    while stack:
        node = stack.pop()
        x, y = node
        if node not in flashed:
            m[x][y] += 1
        else:
            continue

        if m[x][y] > 9:
            flashed.add(node)
            m[x][y] = 0
            stack.extend(list(graph[node]))

    return flashed


if __name__ == "__main__":
    data = []
    with open("input.txt", "r") as f:
        for l in f.readlines():
            data.append([int(c) for c in l.strip()])
    m = np.array(data)

    graph = build_graph(m)

    flash_ct = 0
    for i in range(100):
        flash_ct += len(after_one_step(graph, m))

    print("Answer of AoC 2021 Day 11 Part 1:", flash_ct)


    m = np.array(data)

    graph = build_graph(m)

    n_elements = len(m) * len(m[0])

    n_steps = 0
    while True:
        n_steps += 1
        if len(after_one_step(graph, m)) == n_elements:
            break

    print("Answer of AoC 2021 Day 11 Part 2:", n_steps)
