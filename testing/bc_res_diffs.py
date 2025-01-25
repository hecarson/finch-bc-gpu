import re
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("filename1")
parser.add_argument("filename2")
args = parser.parse_args()
filename1 = args.filename1
filename2 = args.filename2

res_val_re = re.compile(r": (.+)")

with open(filename1, "r") as file1, open(filename2, "r") as file2:
    lines1 = file1.readlines()
    lines2 = file2.readlines()

    for i in range(len(lines1)):
        line1 = lines1[i]; line2 = lines2[i]
        line1 = line1.strip(); line2 = line2.strip()
        match1 = res_val_re.search(line1)
        match2 = res_val_re.search(line2)
        val1 = float(match1[1]); val2 = float(match2[1])
        
        if abs(val1 - val2) > 1e-3:
            print(f"{line1}\t\t{line2}\t\t{val1 - val2}")