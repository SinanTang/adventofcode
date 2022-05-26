from copy import deepcopy
from collections import defaultdict


def one_day_later(all_fish: list):
    new_fish_ls = deepcopy(all_fish)
    for i in range(len(all_fish)):
        if all_fish[i] == 0:
            new_fish_ls[i] = 6
            new_fish_ls.append(8)
        else:
            new_fish_ls[i] -= 1
    return new_fish_ls


def fish_after_n_days_recursive(init_fish: list, n_days: int):
    if n_days == 0:
        return init_fish
    else:
        return one_day_later(fish_after_n_days(init_fish, n_days - 1))


def fish_after_n_days_recursive_memo(fish_dict: dict, n_days: int):
    if n_days == 0:
        return fish_dict[0]
    else:
        if n_days in fish_dict: 
            return fish_dict[n_days]
        else:
            new_fish_ls = one_day_later(fish_after_n_days(fish_dict, n_days - 1))
            fish_dict[n_days] = new_fish_ls

            return new_fish_ls


def fish_after_n_days_iterative(init_fish: list, n_days: int):
    fish_list = init_fish

    i = 0
    while i < n_days:
        print(f"Now computing i = {i}.")
        fish_list = one_day_later(fish_list)
        i += 1

    return fish_list


def fish_after_n_days_efficient(fish_freq: dict, n_days: int):
    for n in range(n_days):
        fish_freq_today = deepcopy(fish_freq)

        for f in fish_freq:
            if f == 0:
                fish_freq_today[8] += fish_freq[f]
                fish_freq_today[6] += fish_freq[f]
                fish_freq_today[0] -= fish_freq[f]
            else:
                fish_freq_today[f] -= fish_freq[f]
                fish_freq_today[f-1] += fish_freq[f]

        fish_freq = fish_freq_today

    return sum(fish_freq.values())


if __name__ == "__main__":
    fish_freq = defaultdict(int)
    with open("input.txt") as f:
        fish = f.read().strip().split(",")
        for fi in fish:
            fish_freq[int(fi)] += 1

    fish_after_80_d = fish_after_n_days_efficient(fish_freq, 80)

    print("Answer of AoC 2021 Day 6 Part 1:", fish_after_80_d)

    fish_after_256_d = fish_after_n_days_efficient(fish_freq, 256)

    print("Answer of AoC 2021 Day 6 Part 2:", fish_after_256_d)
