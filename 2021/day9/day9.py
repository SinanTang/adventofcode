import numpy as np


def is_low_point(m: np.array, x: int, y: int):
    # corners
    if x == 0 and y == 0:
        if m[x][y] < m[0][1] and m[0][0] < m[1][0]:
            return True
    if x == 0 and y == len(m[0]) - 1:
        if m[x][y] < m[x][y-1] and m[x][y] < m[x+1][y]:
            return True
    if x == len(m) - 1 and y == 0:
        if m[x][y] < m[x-1][y] and m[x][y] < m[x][y+1]:
            return True
    if x == len(m) - 1 and y == len(m[0]) - 1:
        if m[x][y] < m[x-1][y] and m[x][y] < m[x][y-1]:
            return True

    # non-edge, non-corner points
    if 0 < x < len(m) - 1 and 0 < y < len(m[0]) - 1:
        if m[x][y] < m[x][y-1] and m[x][y] < m[x][y+1] and m[x][y] < m[x-1][y] and m[x][y] < m[x+1][y]:
            return True
    # edge, non-corner points, first row
    elif x == 0 and 0 < y < len(m[0]) - 1:
        if m[x][y] < m[x][y-1] and m[x][y] < m[x][y+1] and m[x][y] < m[x+1][y]:
            return True
    # edge, non-corner points, last row
    elif x == len(m) - 1 and 0 < y < len(m[0]) - 1:
        if m[x][y] < m[x][y-1] and m[x][y] < m[x][y+1] and m[x][y] < m[x-1][y]:
            return True
    # edge, non-corner points, first col
    elif y == 0 and 0 < x < len(m) - 1:
        if m[x][y] < m[x][y+1] and m[x][y] < m[x+1][y] and m[x][y] < m[x-1][y]:
            return True
    # edge, non-corner points, last col
    elif y == len(m[0]) - 1 and 0 < x < len(m) - 1:
        if m[x][y] < m[x][y-1] and m[x][y] < m[x+1][y] and m[x][y] < m[x-1][y]:
            return True
    return False


def get_low_points(m: np.array):
    low_points = {}

    for idx, h in np.ndenumerate(m):
        row_id, col_id = idx
        if is_low_point(m, row_id, col_id):
            low_points[idx] = h

    return low_points


def calculate_risk_level(low_points: dict):
    return len(low_points) + sum(low_points.values())


def loc_belongs_to_low_point(m: np.array, x: int, y: int, low_points: list) -> tuple:
    # find out which low point location m[x][y] belongs to.
    cur_x, cur_y = x, y

    while 0 <= cur_x < len(m) and 0 <= cur_y < len(m[0]):
        if (cur_x, cur_y) in low_points:
            return (cur_x, cur_y)

        if 0 <= cur_x+1 < len(m) and m[cur_x+1][cur_y] < m[cur_x][cur_y]:
            cur_x = cur_x + 1
            continue

        if 0 <= cur_y+1 < len(m[0]) and m[cur_x][cur_y+1] < m[cur_x][cur_y]:
            cur_y = cur_y + 1
            continue

        if 0 <= cur_y-1 < len(m[0]) and m[cur_x][cur_y-1] < m[cur_x][cur_y]:
            cur_y = cur_y - 1
            continue

        if 0 <= cur_x-1 < len(m) and m[cur_x-1][cur_y] < m[cur_x][cur_y]:
            cur_x = cur_x-1
            continue


def construct_basins(m, low_points):
    basins = {}  # {(0,2): {(0,0), (1,0)}, ...}
    for pt in low_points:
        basins[pt] = set()

    for idx, h in np.ndenumerate(m):
        if h == 9:
            continue
        if idx in low_points.keys():
            continue

        row_id, col_id = idx
        center = loc_belongs_to_low_point(m, row_id, col_id, low_points.keys())
        basins[center].add(idx)

    return basins


def calculate_basins(basins):
    sizes = []
    for b in basins:
        sizes.append(len(basins[b]))
    sizes.sort()
    return (sizes[-1] + 1) * (sizes[-2] + 1) * (sizes[-3] + 1)


if __name__ == "__main__":
    map_ = []
    with open("input.txt") as f:
        for ln in f.readlines():
            map_.append([int(i) for i in ln.strip()])
    m = np.array(map_)

    low_points = get_low_points(m)
    risk_level = calculate_risk_level(low_points)

    print("Answer of AoC 2021 Day 9 Part 1:", risk_level)

    basins = construct_basins(m, low_points)
    answer = calculate_basins(basins)

    print("Answer of AoC 2021 Day 9 Part 2:", answer)
