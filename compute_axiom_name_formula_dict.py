import os
import glob
import pickle
import config
from pathlib import Path

ORIGINAL_DATASET = os.path.join(Path.home(), "gnn-entailment-caption/nndata/")


def main():
    # Get hold of the problem paths
    problems = glob.glob(ORIGINAL_DATASET + "*")
    print(f"Number of problems: {len(problems)}")

    name_axiom_map = {}

    for prob_path in problems:
        with open(prob_path, "r") as f:
            prob = f.readlines()
            for line in prob:
                name = line.split("(", 1)[1].split(",", 1)[0]
                print(name)
                if name not in name_axiom_map:
                    name_axiom_map[name] = line[1:].strip()

    with open(config.axiom_map_path, "wb") as f:
        pickle.dump(name_axiom_map, f)

    print("Number of axioms:", len(name_axiom_map))


if __name__ == "__main__":
    main()
