import numpy as np 


def get_num_ls(inputs):
    return [int(i) for i in inputs[0].split(",")]


def get_boards(inputs):
    boards = []
    
    for i in range(len(inputs[1:]) // 5):
        board = []
        for l in inputs[i * 5 + 1 : i * 5 + 5 + 1]:
            row = [int(n) for n in l.split()]
            board.append(row)
        
        board_ndarray = np.array(board)
        boards.append(board_ndarray)

    return boards


def calculate_win(progress_marker: np.ndarray):

    def row_is_true(row):
        for e in row:
            if e != True:
                return False
        return True

    def col_is_true(col_id):
        for row_id in range(5):
            if progress_marker[row_id][col_id] != True:
                return False
        return True

    for idx in range(5):
        if row_is_true(progress_marker[idx]):
            return True 
        if col_is_true(idx):
            return True

    return False


def board_wins(board: np.ndarray, drawn_nums: list):
    progress_marker = np.full(shape=(5, 5), fill_value=False)

    for idx, n in np.ndenumerate(board):
        row_id, col_id = idx
        if n in drawn_nums:
            progress_marker[row_id][col_id] = True

    return calculate_win(progress_marker)


def calculate_score(board, drawn_nums):
    total = 0
    for num in np.nditer(board):
        if num not in drawn_nums:
            total += num

    return total * drawn_nums[-1]


def play_game(boards, num_ls):
    winning_boards = []
    winning_scores = []

    for i in range(5, len(num_ls)):
        drawn = num_ls[:i]
        # print("Numbers being drawn:", drawn)

        for board_id in range(len(boards)):
            if board_wins(boards[board_id], drawn) and board_id not in winning_boards:
                print(f"Board No. {board_id} just won!")
                winning_boards.append(board_id)
                # print(boards[board_id])
                score = calculate_score(boards[board_id], drawn)
                print("Score of the winning board:", score)
                winning_scores.append(score)
    
    return winning_boards, winning_scores


if __name__ == "__main__":
    inputs = []
    with open("input.txt") as f:
        for l in f.readlines():
            if len(l.strip()) > 0:
                inputs.append(l.strip())

    num_ls = get_num_ls(inputs)

    boards = get_boards(inputs)

    winning_boards, winning_scores = play_game(boards, num_ls)

    print("Answer of AoC 2021 Day 4 Part 1:", winning_scores[0])

    print("Answer of AoC 2021 Day 4 Part 2:", winning_scores[-1])
