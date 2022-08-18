import os
import glob
import pickle
from pathlib import Path

PROBLEMS = os.path.join(Path.home(), "gnn-entailment-caption/nndata/")


def main():
    # Get hold of the problem paths
    problems = glob.glob(PROBLEMS + "*")
    print(f"Number of problems: {len(problems)}")

    name_axiom_map = {}

    for prob_path in problems:
        with open(prob_path, "r") as f:
            prob = f.readlines()
            for line in prob:
                name = line.split("(", 1)[1].split(",", 1)[0]
                if name not in name_axiom_map:
                    name_axiom_map[name] = line[1:].strip()

    with open("data/name_axiom_map.pkl", "wb") as f:
        pickle.dump(name_axiom_map, f)


if __name__ == "__main__":
    main()
