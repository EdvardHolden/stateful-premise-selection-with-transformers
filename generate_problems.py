import torch
import torch.nn as nn
import pickle
import os
from tqdm import tqdm

import config
from config import model_params
from transformer import Transformer
from dataset import StatementDataset
from torch.utils.data import DataLoader
from collate import VarLengthCollate
from predict import build_vocabs


def get_problem_conjecture(problem_name):

    # Get problem name
    prob_path = os.path.join(config.ORIGINAL_DATASET, problem_name)

    # Get the first line - contains the conjecture
    with open(prob_path, "r") as f:
        conj = f.readline()

    conj = conj[1:].replace("axiom", "conjecture", 1)
    return conj


def main():

    # Build vocabulary
    source_vocab, target_vocab = build_vocabs(config.train_source_path, config.train_target_path)

    # Build dataloader for the prediction set
    test_dataset = StatementDataset(config.evaluate_source_path, source_vocab, model_params["conjecture_max_length"])
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=10,
        shuffle=False,
        collate_fn=VarLengthCollate(batch_dim=0),
        pin_memory=True,
    )

    # Initialise transformer model
    model = Transformer(
        source_vocab,
        target_vocab,
        model_params["model_num_layers"],
        model_params["model_state_size"],
        model_params["model_num_heads"],
        model_params["model_hidden_size"],
        model_params["conjecture_max_length"],
        target_position=False,
    )

    # Load the weights of the model
    model.load_state_dict(torch.load(config.model_path))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(config.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameters {}".format(num_params))

    # Load problem names to map with the conjecture and name
    with open(os.path.join(os.path.dirname(config.evaluate_source_path), "ids"), "r") as f:
        problem_names = f.read().splitlines()
    assert len(problem_names) == len(test_data_loader)
    print("Number of problems:", len(problem_names))

    # Load axiom map path
    with open(config.axiom_map_path, "rb") as f:
        name_axiom_map = pickle.load(f)

    for prob_no, (prob_name, prob_x) in tqdm(enumerate(zip(problem_names, test_data_loader))):

        x = prob_x["source"].to(config.device)
        # Return n number of beams
        output = model(source=x, type="beam_search", beam_size=10, max_length=64)
        # Collect all beams into a single set - but only on axioms names
        predicted_axioms = set()
        for indices in output:
            premises = []
            for index in indices:
                if index != config.START_END_INDEX:
                    if index != config.PAD_INDEX:
                        premises.append(target_vocab.index2word[index.item()])
                else:
                    break
            predicted_axioms = predicted_axioms.union(set(premises))

        # Map the axiom names to formulae
        predicted_axioms = [name_axiom_map[name] for name in predicted_axioms]

        # Get hold of the conjecture
        conj = get_problem_conjecture(prob_name)

        # Write to output file
        with open("./generated_problems/" + str(prob_name), "w") as f:
            f.write(conj.strip() + "\n")
            f.write("\n".join(predicted_axioms))


if __name__ == "__main__":
    main()
