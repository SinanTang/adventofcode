# test_depths = [199, 200, 208, 210, 200, 207, 240, 269, 260, 263]

depths = []
with open("input.txt") as f:
    for i in f.readlines():
        depths.append(int(i.strip()))


def compare_measurements(depths):
    increased, decreased, same = 0, 0, 0
    for i in range(1, len(depths)):
        if depths[i] > depths[i-1]: increased += 1
        if depths[i] < depths[i-1]: decreased += 1
        if depths[i] == depths[i-1]: same += 1
    return increased, decreased, same
    

def get_sums(depths):
    sums = []
    for i in range(len(depths) - 2):
        sum_ = depths[i] + depths[i+1] + depths[i+2]
        sums.append(sum_)
    return sums


def part_1():
    increased, decreased, same = compare_measurements(depths)
    return increased


def part_2():
    sums = get_sums(depths)
    increased, decreased, same = compare_measurements(sums)
    return increased


if __name__ == "__main__":
    print("Answer of AoC 2021 Day 1 Part 1:", part_1())
    
    print("Answer of AoC 2021 Day 1 Part 2:", part_2())
