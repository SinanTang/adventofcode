from collections import defaultdict


def build_graph(connections, caves):
    '''
    graph = {
        "start": set("A", "b"),
        "A": set("c", "b", "end"),
        "b": set("A", "d", "end"),
        "d": set("b"),
        "c": set("A")
    }
    '''
    graph = defaultdict(set)
    for c in connections:
        if c[0] in caves:
            if c[1] != "start":
                graph[c[0]].add(c[1])
        elif c[0] == "start":
            graph["start"].add(c[1])
        # repeat for the second element
        if c[1] in caves:
            if c[0] != "start":
                graph[c[1]].add(c[0])
        elif c[1] == "start":
            graph["start"].add(c[0])
    return graph


def separate_caves(caves):
    big_caves, small_caves = [], []
    for c in caves:
        if c.lower() == c:
            small_caves.append(c)
        else:
            big_caves.append(c)
    return big_caves, small_caves


def find_paths_part_1(graph, caves, start="start", goal="end"):
    _, small_caves = separate_caves(caves)
    # bfs
    queue = [(start, [start])]

    while queue:
        node, path = queue.pop(0)

        for next_ in graph[node]:
            if next_ in small_caves and next_ in path:
                continue
            if next_ == goal:
                yield path + [next_]
            else:
                queue.append((next_, path + [next_]))


def small_cave_visited_twice(path, small_caves):
    small_caves_dict = defaultdict(int)
    for loc in path:
        if loc in small_caves:
            small_caves_dict[loc] += 1

    if 2 in small_caves_dict.values():
        return True
    return False


def find_paths_part_2(graph, caves, start="start", goal="end"):
    _, small_caves = separate_caves(caves)
    # bfs
    queue = [(start, [start])]

    while queue:
        node, path = queue.pop(0)
        for next_ in graph[node]:
            if next_ in small_caves and next_ in path:
                if not small_cave_visited_twice(path, small_caves):
                    queue.append((next_, path + [next_]))
                else:
                    continue
            elif next_ == goal:
                yield path + [next_]
            else:
                queue.append((next_, path + [next_]))


if __name__ == "__main__":
    connections, caves = [], set()
    with open("input.txt", "r") as f:
        for l in f.readlines():
            locs = l.strip().split("-")
            connections.append(tuple(locs))

            for loc in locs:
                if not (loc == "start" or loc == "end"):
                    caves.add(loc)


    graph = build_graph(connections, caves)

    all_paths_1 = []
    for p in find_paths_part_1(graph, caves):
        all_paths_1.append(p)

    print("Answer of AoC 2021 Day 12 Part 1:", len(all_paths_1))


    all_paths_2 = []
    for p in find_paths_part_2(graph, caves):
        all_paths_2.append(p)

    print("Answer of AoC 2021 Day 12 Part 2:", len(all_paths_2))
