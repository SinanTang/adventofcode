# test_inputs = ["forward 5", "down 5", "forward 8", "up 3", "down 8", "forward 2"]

inputs = []
with open("input.txt") as f:
    for l in f.readlines():
        inputs.append(l.strip())


def calculate_commands_1(inputs):        
    pos, depth = 0, 0
    for i in inputs:
        direction, distance = i.split(" ")
        if direction == "forward":
            pos += int(distance)
        elif direction == "down":
            depth += int(distance)
        elif direction == "up":
            depth -= int(distance)

    return pos, depth


def calculate_commands_2(inputs):
    pos, depth, aim = 0, 0, 0
    for i in inputs:
        direction, distance = i.split(" ")
        if direction == "forward":
            pos += int(distance)
            depth += int(distance) * aim
        elif direction == "down":
            aim += int(distance)
        elif direction == "up":
            aim -= int(distance)

    return pos, depth, aim
    

if __name__ == "__main__":
    pos_1, depth_1 = calculate_commands_1(inputs)
    print("Answer of AoC 2021 Day 2 Part 1:", pos_1 * depth_1)

    pos_2, depth_2, aim = calculate_commands_2(inputs)
    print("Answer of AoC 2021 Day 2 Part 2:", pos_2 * depth_2)
