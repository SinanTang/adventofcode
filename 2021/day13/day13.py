import numpy as np


def construct_origami_paper(dots: list) -> np.array:
    # dots e.g. [(6, 10), (0, 14), (9, 10), ...]
    dots_x = [d[0] for d in dots]
    dots_y = [d[1] for d in dots]
    max_x, max_y = max(dots_x), max(dots_y)
    # definition of x / y here is opposite to the usual np array definition
    m = np.full(shape=(max_y + 1, max_x + 1), fill_value=".")

    for dot in dots:
        x, y = dot[0], dot[1]
        m[y][x] = "#"

    return m


def fold(paper: np.array, instruction: tuple) -> np.array:
    fold_aixs, fold_idx = instruction

    if fold_aixs == "y":
        max_x = fold_idx
        folded_paper = paper[:max_x,]
        for idx, val in np.ndenumerate(paper[:max_x,]):
            x, y = idx
            x_ = fold_idx * 2 - x 
            try:
                if paper[x][y] == "." and paper[x_][y] == "#": 
                    folded_paper[x][y] = "#"
            except IndexError:
                continue

    elif fold_aixs == "x":
        max_y = fold_idx
        folded_paper = paper[:,:max_y]
        for idx, val in np.ndenumerate(paper[:,:max_y]):
            x, y = idx 
            y_ = fold_idx * 2 - y
            try:
                if paper[x][y] == "." and paper[x][y_] == "#": 
                    folded_paper[x][y] = "#"
            except IndexError:
                continue

    return folded_paper


def count_dots(paper: np.array) -> int:
    c = 0
    for val in np.nditer(paper):
        if val == "#":
            c += 1
    return c


if __name__ == "__main__":
    origami_dots, instructions = [], []
    with open("input.txt") as f:
        for ln in f.readlines():
            ln = ln.strip()
            if ln.startswith("fold along"):
                fold_ln = ln.split(" ")[-1]
                instructions.append(
                    (fold_ln.split("=")[0], int(fold_ln.split("=")[1]))
                    )
            elif len(ln) > 0:
                origami_dots.append(tuple([int(c) for c in ln.split(",")]))

    origami_paper = construct_origami_paper(origami_dots)

    first_fold = fold(origami_paper, instructions[0])
    visible_dots = count_dots(first_fold)

    print("Answer of AoC 2021 Day 13 Part 1:", visible_dots)

    paper = origami_paper
    for i in instructions:
        folded_paper = fold(paper, i)
        paper = folded_paper
    
    with open("output.txt", "w") as f:
        for lst in paper.tolist():
            ln = ''.join(lst)
            f.write(ln+"\n")

    # print("Answer of AoC 2021 Day 13 Part 2:") # REUPUPKR
