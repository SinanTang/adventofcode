# test_inputs = ["00100", "11110", "10110", "10111", "10101", "01111", "00111", "11100", "10000", "11001", "00010", "01010"]

def most_common_b(lst):
    c = Counter(lst)
    return c.most_common(1)[0][0]


def least_common_b(lst):
    c = Counter(lst)
    return c.most_common()[-1][0]


def count_is_equal(num_ls, pos):
    bit_ls = [n[pos] for n in num_ls]
    c = Counter(bit_ls).most_common()
    if c[0][1] == c[1][1]:
        return True
    return False


def get_oxygen_generator_rating(inputs):
    r = {}
    total_pos = len(inputs[0])

    for pos in range(total_pos):
        if pos > 0:
            prev_num_ls = r[pos - 1]
            if count_is_equal(prev_num_ls, pos):
                num_ls = [n for n in prev_num_ls if n[pos] == "1"]
            else:
                bits = [n[pos] for n in prev_num_ls]
                most_common_bit = most_common_b(bits)
                num_ls = [n for n in prev_num_ls if n[pos] == most_common_bit]

            num_ls = [n for n in num_ls if n in prev_num_ls]

        elif pos == 0:
            if count_is_equal(inputs, pos):
                num_ls = [n for n in inputs if n[pos] == "1"]
            else:
                bits = [n[pos] for n in inputs]
                most_common_bit = most_common_b(bits)
                num_ls = [n for n in inputs if n[pos] == most_common_bit]

        if len(num_ls) == 1:
            return num_ls[0]

        r[pos] = num_ls

    return "Didn't find the rating :("


def get_co2_scrubber_rating(inputs):
    r = {}
    total_pos = len(inputs[0])

    for pos in range(total_pos):
        if pos > 0:
            prev_num_ls = r[pos - 1]
            if count_is_equal(prev_num_ls, pos):
                num_ls = [n for n in prev_num_ls if n[pos] == "0"]
            else:
                bits = [n[pos] for n in prev_num_ls]
                least_common_bit = least_common_b(bits)
                num_ls = [n for n in prev_num_ls if n[pos] == least_common_bit]

            num_ls = [n for n in num_ls if n in prev_num_ls]

        elif pos == 0:
            if count_is_equal(inputs, pos):
                num_ls = [n for n in inputs if n[pos] == "0"]
            else:
                bits = [n[pos] for n in inputs]
                least_common_bit = least_common_b(bits)
                num_ls = [n for n in inputs if n[pos] == least_common_bit]

        if len(num_ls) == 1:
            return num_ls[0]

        r[pos] = num_ls

    return "Didn't find the rating :("


def get_gamma_rate(inputs):
    r = []
    len_ = len(inputs[0])

    for d in range(len_):
        bits = [inputs[i][d] for i in range(len(inputs))]
        most_common_bit = most_common_b(bits)
        r.append(most_common_bit)
    
    return r


def get_epsilon_rate(gamma_rate):
    return ["1" if i == "0" else "0" for i in gamma_rate]


def get_decimal(lst):
    s = ''.join(lst)
    return int(s, 2)


if __name__ == "__main__":
    inputs = []
    with open("input.txt") as f:
    for i in f.readlines():
        inputs.append(int(i.strip()))

    gamma_lst = get_gamma_rate(inputs)
    epsilon_lst = get_epsilon_rate(inputs)

    print("Answer of AoC 2021 Day 3 Part 1:", get_decimal(gamma_lst) * get_decimal(epsilon_lst))

    oxygen_generator_rating = get_oxygen_generator_rating(inputs)
    co2_scrubber_rating = get_co2_scrubber_rating(inputs)

    print("Answer of AoC 2021 Day 3 Part 2:", int(oxygen_generator_rating, 2) * int(co2_scrubber_rating, 2))
