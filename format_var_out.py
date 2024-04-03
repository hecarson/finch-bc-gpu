

while True:
    try:
        line = input()
    except EOFError:
        break

    if line == "" or line[0] != '[':
        print(line)
        continue

    array_lines = line.split(';')