# Split each line containing a global vector into lines, where each line has values for a node

from argparse import ArgumentParser

NUM_DIRS = 4
NUM_BANDS = 13
num_vals_per_node = NUM_DIRS * NUM_BANDS

parser = ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()
filename = args.filename
out_filename = filename.split(".txt")[0] + "-format.txt"

with open(filename, "r") as file, open(out_filename, "w") as out_file:
    while True:
        line = file.readline()
        if line == "":
            break

        if line == '\n' or line[0] != '[':
            line = line.strip()
            print(line, file=out_file)
            continue
        
        line = line[1:-2]
        vals = line.split(", ")
        num_nodes = len(vals) // num_vals_per_node
        
        for node_idx in range(num_nodes):
            node_vals = vals[node_idx * num_vals_per_node : (node_idx + 1) * num_vals_per_node]
            line = ", ".join(node_vals)
            print(line, file=out_file)