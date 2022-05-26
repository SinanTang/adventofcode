from copy import deepcopy
import numpy as np


def mark_points_in_diagram(coordinates: list[tuple], 
    diagram: np.ndarray, 
    consider_diagonal: bool = False):

    all_points = deepcopy(coordinates)

    x1, y1 = coordinates[0]
    x2, y2 = coordinates[1]
    # conform to artificial rule of x1 >= x2 for simpler computation later
    if x1 < x2:
        temp_x, temp_y = x1, y1
        x1, y1 = x2, y2
        x2, y2 = temp_x, temp_y

    if x1 == x2:
        for y in range(min(y1, y2) + 1, max(y1, y2)):
            all_points.append((x1, y))
    elif y1 == y2:
        for x in range(x2 + 1, x1):
            all_points.append((x, y1))
    elif consider_diagonal is False:
        return
    else:
        if x1 - x2 == y1 - y2:
            for x in range(x2 + 1, x1):
                all_points.append((x, y2 + (x - x2)))
        else:
            for x in range(x2 + 1, x1):
                all_points.append((x, y2 - (x - x2)))

    for p in set(all_points):
        x, y = p[0], p[1]
        diagram[y][x] += 1
    


def calculate_points(diagram):
    total = 0
    for i in np.nditer(diagram):
        if i >= 2:
            total += 1

    return total


if __name__ == "__main__":
    inputs = []
    max_x, max_y = 0, 0

    with open("input.txt") as f:
        for l in f.readlines():
            coordinates = []
            l = l.strip()
            for i in l.split(" -> "):
                pts = i.split(",")
                
                x, y = int(pts[0]), int(pts[1])
                if x > max_x: max_x = x
                if y > max_y: max_y = y
                
                coordinates.append((x, y))

            inputs.append(coordinates)

    diagram = np.zeros((max_y + 1, max_x + 1), dtype=int)

    for coordinates in inputs:
        mark_points_in_diagram(coordinates, diagram)

    num_points = calculate_points(diagram)

    print("Answer of AoC 2021 Day 5 Part 1:", num_points)


    diagram_2 = np.zeros((max_y + 1, max_x + 1), dtype=int)

    for coordinates in inputs:
        mark_points_in_diagram(coordinates, diagram_2, consider_diagonal=True)

    num_points_2 = calculate_points(diagram_2)

    print("Answer of AoC 2021 Day 5 Part 2:", num_points_2)
