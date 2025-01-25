"""
Finds the maximum difference between two solution arrays.

Example:
python check_sol_error.py ad2d/{fvad2d-sol.txt,fvad2dgpu-sol.txt}
"""

import re
from argparse import ArgumentParser

def get_var_vals(filename: str) -> list[float]:
    with open(filename, "r") as file:
        arr_str = file.read()
    arr_str = arr_str.strip()
    arr_start_idx = arr_str.find('[')
    arr_end_idx = arr_str.find(']')
    arr_str = arr_str[arr_start_idx + 1 : arr_end_idx]
    
    vals = [float(v) for v in re.split(r" |; ", arr_str)]
    return vals

parser = ArgumentParser()
parser.add_argument("filename1")
parser.add_argument("filename2")
args = parser.parse_args()
filename1 = args.filename1
filename2 = args.filename2

vals1 = get_var_vals(filename1)
vals2 = get_var_vals(filename2)

errors = []
for i in range(len(vals1)):
    error = abs(vals1[i] - vals2[i]) / abs(vals1[i])
    errors.append(error)
print(errors)
print(f"max error {max(errors)}")