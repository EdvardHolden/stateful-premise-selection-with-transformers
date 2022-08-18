import numpy as np
import torch
import torch.nn as nn

import config
from config import model_params
from transformer import Transformer
from dataset import StatementDataset
from torch.utils.data import DataLoader
from collate import VarLengthCollate
from vocabulary import SourceVocabulary, TargetVocabulary


def subsequent_mask(size):
    shape = (1, size, size)
    mask = np.triu(np.ones(shape), k=1).astype(np.uint8)
    return torch.from_numpy(mask) == 0


def data_gen(high, batch_size, num_batches):
    for i in range(num_batches):
        data = torch.from_numpy(np.random.randint(2, high, size=(batch_size, 12)))
        data[:, 0] = 1
        data[:, -1] = 1
        source = data
        target = data.flip(1)
        # target_in = target[:, :-1]
        # target_out = target[:, 1:]
        # source_mask = (source != 0).unsqueeze(1)
        # target_mask = (target_in != 0).unsqueeze(1) & subsequent_mask(target_in.size(-1))
        yield source, target


def build_vocabs(source_path, target_path):
    source_vocab = SourceVocabulary()
    target_vocab = TargetVocabulary()
    with open(source_path, "r") as source_file, open(target_path, "r") as target_file:
        source_lines = []
        target_lines = []
        while True:
            source_line = source_file.readline().strip("\n")
            target_line = target_file.readline().strip("\n")
            if (not source_line) or (not target_line):
                break
            source_lines.append(source_line)
            target_lines.append(target_line)
            source_vocab.add_sentence(source_line)
            target_vocab.add_sentence(target_line)
    return source_vocab, target_vocab


def get_union_and_intersection_size(output, premises):
    union, counts = torch.cat([output, premises]).unique(return_counts=True)
    union_size = float(len(union))
    intersection_size = (counts > 1).sum().item()
    num_premises = float(len(premises))
    # jaccard = (intersection_size / union_size).item()
    # coverage = (intersection_size / num_premises).item()
    # print(output, premises)
    # print(union, counts)
    return union_size, intersection_size, num_premises


def main():

    # Build vocabulary
    source_vocab, target_vocab = build_vocabs(config.train_source_path, config.train_target_path)

    # Build dataloader for the prediction set
    test_dataset = StatementDataset(config.evaluate_source_path, source_vocab)
    test_data_loader = DataLoader(
        test_dataset, batch_size=1, num_workers=10, collate_fn=VarLengthCollate(batch_dim=0), pin_memory=True
    )

    # Initialise transformer model
    model = Transformer(
        source_vocab,
        target_vocab,
        model_params["model_num_layers"],
        model_params["model_state_size"],
        model_params["model_num_heads"],
        model_params["model_hidden_size"],
        target_position=False,
    )

    # Load the weights of the model
    model.load_state_dict(torch.load(config.model_path))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(config.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameters {}".format(num_params))

    all_premises = []
    for batch in test_data_loader:
        x = batch["source"].to(config.device)
        output = model(source=x, type="beam_search", beam_size=10, max_length=64)
        top_premises = []
        for indices in output:
            premises = []
            for index in indices:
                if index != target_vocab.START_END_INDEX:
                    premises.append(target_vocab.index2word[index.item()])
                else:
                    break
            premises = " ".join(premises)
            top_premises.append(premises)
        top_premises = "\n".join(top_premises)
        all_premises.append(top_premises)
    all_premises = "\n".join(all_premises)
    with open(config.predictions_path, "w") as predictions_file:
        predictions_file.write(all_premises)


if __name__ == "__main__":
    main()
