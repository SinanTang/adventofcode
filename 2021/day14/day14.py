from collections import defaultdict


def one_step_later(polymer_pairs: dict, rules: dict) -> dict:
    new_polymer = defaultdict(int)

    for pair in polymer_pairs:
        new_polymer_bit = defaultdict(int)
        if pair in rules:
            new_1, new_2 = f"{pair[0]}{rules[pair]}", f"{rules[pair]}{pair[1]}"
            new_polymer_bit[new_1] += polymer_pairs[pair]
            new_polymer_bit[new_2] += polymer_pairs[pair]
        else:
            new_polymer_bit[pair] = polymer_pairs[pair]

        for p in new_polymer_bit:
            new_polymer[p] += new_polymer_bit[p]

    return new_polymer
            

def calculate_elements(polymer_pairs: dict, init_polymer: str) -> dict:
    elements = defaultdict(int)

    elements[init_polymer[0]] = 1
    elements[init_polymer[-1]] = 1

    for pair in polymer_pairs:
        elements[pair[0]] += polymer_pairs[pair]
        elements[pair[1]] += polymer_pairs[pair]

    return {k: v//2 for k, v in elements.items()}


def calculate_result(polymer_in_elements: dict) -> int:
    return max(polymer_in_elements.values()) - min(polymer_in_elements.values())


def polymer_in_pairs(polymer: str) -> dict:
    pairs = defaultdict(int)
    for i in range(len(polymer) - 1):
        pairs[polymer[i:i+2]] += 1
    return pairs


def parse_input(input_filename: str) -> tuple[str, dict]:
    rules = {}
    with open(input_filename, "r") as f:
        lines = f.read().split("\n\n")
        init_polymer = lines[0]

        for r in lines[1].split("\n"):
            if len(r) > 1:
                rule = r.split(" -> ")
                rules[rule[0]] = rule[1]

    return init_polymer, rules


if __name__ == "__main__":
    init_polymer, rules = parse_input("input.txt")

    polymer_pair = polymer_in_pairs(init_polymer)
    for i in range(10):
        polymer_pair = one_step_later(polymer_pair, rules)

    elements_after_10_steps = calculate_elements(polymer_pair, init_polymer)
    result_part_1 = calculate_result(elements_after_10_steps)

    print("Answer of AoC 2021 Day 14 Part 1:", result_part_1)


    polymer_pair = polymer_in_pairs(init_polymer)
    for i in range(40):
        polymer_pair = one_step_later(polymer_pair, rules)

    elements_after_40_steps = calculate_elements(polymer_pair, init_polymer)
    result_part_2 = calculate_result(elements_after_40_steps)
    
    print("Answer of AoC 2021 Day 14 Part 2:", result_part_2)

