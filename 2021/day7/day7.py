from collections import defaultdict


def get_constant_fuel_consumption_to_position(pos, crabs):
    total_fuel = 0
    for crab in crabs:
        f = abs(pos - crab)
        total_fuel += f * crabs[crab]
    return total_fuel


def get_linear_fuel_consumption_to_position(pos, crabs):
    total_fuel = 0
    for crab in crabs:
        n = abs(pos - crab)
        f = (1 + n) * n // 2
        total_fuel += f * crabs[crab]
    return total_fuel


def best_position_part1(crabs):
    min_fuel, min_pos = float("inf"), None

    for pos in range(min(crabs.keys()), max(crabs.keys())):
        f = get_constant_fuel_consumption_to_position(pos, crabs)
        if f < min_fuel:
            min_fuel, min_pos = f, pos
    
    return min_fuel, min_pos


def best_position_part2(crabs):
    min_fuel, min_pos = float("inf"), None

    for pos in range(min(crabs.keys()), max(crabs.keys())):
        f = get_linear_fuel_consumption_to_position(pos, crabs)
        if f < min_fuel:
            min_fuel, min_pos = f, pos

    return min_fuel, min_pos


if __name__ == "__main__":
    crab_freq = defaultdict(int)
    with open("input.txt") as f:
        crabs = f.read().strip().split(",")
        for c in crabs:
            crab_freq[int(c)] += 1

    min_fuel, min_pos = best_position_part1(crab_freq)

    print("Answer of AoC 2021 Day 7 Part 1:", min_fuel)

    min_fuel, min_pos = best_position_part2(crab_freq)

    print("Answer of AoC 2021 Day 7 Part 2:", min_fuel)
