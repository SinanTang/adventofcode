brackets = {"(": ")", "{": "}", "[": "]", "<": ">"}


def find_illegal_char(line):
    stack = []
    for c in line:
        if c in brackets:
            stack.append(c)
        else:
            if len(stack) > 0 and brackets[stack[-1]] == c:
                stack.pop()
            else:
                # print(f"Found illegal char: {c}\n")
                return c
    # print("No illegal char in line.\n")


def calculate_score_part_1(lines):
    illegal_chars = {}
    for line in lines:
        result = find_illegal_char(line)
        if result:
            illegal_chars[result] = illegal_chars.get(result, 0) + 1

    score = 0
    for char in illegal_chars:
        if char == ")":
            score += illegal_chars[char] * 3
        elif char == "]":
            score += illegal_chars[char] * 57
        elif char == "}":
            score += illegal_chars[char] * 1197
        elif char == ">":
            score += illegal_chars[char] * 25137
    return score


def unmatched_chars(line):
    stack = []
    for c in line:
        if c in brackets:
            stack.append(c)
        else:
            if len(stack) > 0 and brackets[stack[-1]] == c:
                stack.pop()
            else:
                # contains illegal char
                return (False, stack)
    # doesn't contain illegal char, might be incomplete
    return (True, stack)


def auto_complete_brackets(left_brackets):
    right_brackets = []
    for i in left_brackets[::-1]:
        right_brackets.append(brackets[i])
    return right_brackets


def calculate_score_part_2(chars):
    add_value = {")": 1, "]": 2, "}": 3, ">": 4}
    score = 0
    for char in chars:
        score *= 5
        score += add_value[char]
    return score


def get_middle_score_part_2(lines):
    scores = []
    for i in range(len(lines)):
        has_no_illegal_char, remaining_chars = unmatched_chars(lines[i])
        if has_no_illegal_char and len(remaining_chars) > 0:
            chars_to_add = auto_complete_brackets(remaining_chars)
            score = calculate_score_part_2(chars_to_add)
            scores.append(score)

    scores.sort()
    return scores[len(scores)//2]


if __name__ == "__main__":
    with open("input.txt", "r") as f:
        lines = [l.strip() for l in f.readlines()]

    score = calculate_score_part_1(lines)

    print("Answer of AoC 2021 Day 10 Part 1:", score)


    middle_score = get_middle_score_part_2(lines)

    print("Answer of AoC 2021 Day 10 Part 2:", middle_score)
