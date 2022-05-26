# num of segments -> digit candidates
segments = {2: 1, 3: 7, 4: 4, 5: [2, 3, 5], 6: [0, 6, 9], 7: 8}


def part_1_solution(signals):
    c = 0
    for signal in signals:
        output_vals = signal[-4:]
        for v in output_vals:
            if len(v) == 2 or len(v) == 3 or len(v) == 4 or len(v) == 7:
                c += 1
    return c


def decode_one_display(signal: list):
    digit_char = {}
    input_vals, output_vals = signal[:-5], signal[-4:]

    for v in input_vals:
        if len(v) == 2:
            digit_char[segments[2]] = v
        elif len(v) == 3:
            digit_char[segments[3]] = v
        elif len(v) == 4:
            digit_char[segments[4]] = v
        elif len(v) == 7:
            digit_char[segments[7]] = v

    for v in input_vals:
        if len(v) == 5:
            if digit_char[1][0] in v and digit_char[1][1] in v:
                digit_char[3] = v
            elif len(set(digit_char[4]) & set(v)) == 3:
                digit_char[5] = v
            else:
                digit_char[2] = v
        elif len(v) == 6: 
            if len(set(digit_char[4]) & set(v)) == segments[4]:
                digit_char[9] = v
            elif digit_char[1][0] in v and digit_char[1][1] in v:
                digit_char[0] = v
            else:
                digit_char[6] = v

    output_s = ''
    for o in output_vals:
        # 1, 4, 7, 8
        if len(o) == 2 or len(o) == 3 or len(o) == 4 or len(o) == 7:
            output_s += str(segments[len(o)])
        # 2, 3, 5
        elif len(o) == 5:
            for candidate in segments[5]:
                if len(set(o) & set(digit_char[candidate])) == 5:
                    output_s += str(candidate)
        # 0, 6, 9
        elif len(o) == 6:
            for candidate in segments[6]:
                if len(set(o) & set(digit_char[candidate])) == 6:
                    output_s += str(candidate)
        else:
            print("Unrecognized input:", o)

    return int(output_s)


def part_2_solution(signals):
    total_sum = 0

    for s in signals:
        total_sum += decode_one_display(s)

    return total_sum


if __name__ == "__main__":
    signals = []
    with open("input.txt") as f:
        for ln in f.readlines():
            signals.append(ln.strip().split(" "))

    ct = part_1_solution(signals)

    print("Answer of AoC 2021 Day 8 Part 1:", ct)

    sum_ = part_2_solution(signals)
    
    print("Answer of AoC 2021 Day 8 Part 2:", sum_)
