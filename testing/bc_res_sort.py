import re
from argparse import ArgumentParser

MAX_BI = 4
MAX_FI = 5
MAX_BAND = 13
MAX_DIR = 4

parser = ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()
filename = args.filename
out_filename = filename.split(".txt")[0] + "-sorted.txt"

res_idx_re = re.compile(r"bi (\d+) fi (\d+) band (\d+) dir (\d+)")
res_val_re = re.compile(r": (.+)")

with open(filename, "r") as file, open(out_filename, "w") as out_file:
    lines = file.readlines()

    for i in range(len(lines)):
        line = lines[i]
        match = res_val_re.search(line)
        val = float(match[1])
        lines[i] = line.split(':')[0] + f": {val}\n"

    def _get_line_key(line: str) -> int:
        match = res_idx_re.match(line)
        bi = int(match[1])
        fi = int(match[2])
        band = int(match[3])
        dir = int(match[4])
        res_idx = (dir - 1) + MAX_DIR * (band - 1) + MAX_DIR * MAX_BAND * (fi - 1) + MAX_DIR * MAX_BAND * MAX_FI * (bi - 1)
        return res_idx

    lines.sort(key=_get_line_key)

    out_file.writelines(lines)