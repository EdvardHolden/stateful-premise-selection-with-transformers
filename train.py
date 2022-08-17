import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import config
from collate import VarLengthCollate
from vocabulary import SourceVocabulary, TargetVocabulary
from dataset import Statement2PremisesDataset
from transformer import Transformer
from learning_rate import NoamOpt

# Some model parameters
model_num_layers = 3
model_state_size = 512
model_num_heads = 8
model_hidden_size = 2048


# Some training parameters
ACCUM_COUNT = 2  # Parameter for how often the weights are updates during training - a bit strange
cross_entropy_loss = nn.NLLLoss(ignore_index=config.PAD_INDEX, reduction="sum")


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


def train_step(model, train_data_loader, step_count, optimizer):

    model.train()
    train_loss = 0
    num_words = 0
    for batch_no, batch in enumerate(train_data_loader):
        print(f"Batch: {batch_no}")

        # Transform input
        x = batch["source"].to(config.device)
        y = batch["target"].to(config.device)
        y_in = y[:, :-1]
        y_out = y[:, 1:]

        # Pass data and compute loss
        output = model(x, y_in)
        mask = y_out != config.PAD_INDEX
        loss = cross_entropy_loss(output.view(-1, output.size()[2]), y_out.flatten())
        # loss = label_smoothing_loss(output.view(-1, target_vocab.num_words), y_out.flatten())

        # Update loss trackers
        train_loss += loss.item()
        num_words += mask.sum().item()
        loss.backward()

        # Apply optimizer
        if step_count % ACCUM_COUNT == 0:
            optimizer.step()
            optimizer.zero_grad()

        step_count += 1

    # Compute final loss
    train_loss /= num_words

    return model, train_loss, step_count


def evaluate_step(model, valid_data_loader):

    model.eval()
    avg_jaccard = 0
    avg_coverage = 0

    with torch.no_grad():
        print("Evaluating...")
        for batch in valid_data_loader:
            x = batch["source"].to(config.device)
            y = batch["target"].to(config.device)
            premises = y[:, 1:]
            premises = premises[(premises != config.PAD_INDEX) & (premises != config.START_END_INDEX)]
            output = model(source=x, type="beam_search", beam_size=10, max_length=64)
            output = output[(output != config.PAD_INDEX) & (output != config.START_END_INDEX)].unique()
            union_size, intersection_size, num_premises = get_union_and_intersection_size(output, premises)
            avg_jaccard += intersection_size / union_size
            avg_coverage += intersection_size / num_premises

        avg_jaccard /= len(valid_data_loader)  # This yields the number of batches, but the batch size is one.
        avg_coverage /= len(valid_data_loader)

    return avg_jaccard, avg_coverage


def main():

    # Build training dataset
    source_vocab, target_vocab = build_vocabs(config.train_source_path, config.train_target_path)

    # Build vocabulary
    train_dataset = Statement2PremisesDataset(
        config.train_source_path, config.train_target_path, source_vocab, target_vocab
    )
    print("Train size {}".format(len(train_dataset)))
    # Build train dataloader
    train_data_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        num_workers=8,
        collate_fn=VarLengthCollate(batch_dim=0),
        pin_memory=True,
    )

    # Build validation dataset
    valid_dataset = Statement2PremisesDataset(
        config.valid_source_path, config.valid_target_path, source_vocab, target_vocab
    )
    print("Valid size {}".format(len(valid_dataset)))

    # Build validation data loader - bathc size of one
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=1, num_workers=10, collate_fn=VarLengthCollate(batch_dim=0), pin_memory=True
    )

    # Initialise transformer model
    model = Transformer(
        source_vocab,
        target_vocab,
        model_num_layers,
        model_state_size,
        model_num_heads,
        model_hidden_size,
        target_position=False,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameters {}".format(num_params))

    # Run in parallel if multiple devices
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(config.device)

    # Initialise the optimiser
    optimizer = NoamOpt(
        model_state_size, 2, 8000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.998), eps=1e-9)
    )

    step_count = 1
    for epoch in range(1, config.EPOCHS):

        # Train step
        model, train_loss, step_count = train_step(model, train_data_loader, step_count, optimizer)

        # Evaluate step
        if config.EVALUATE:
            avg_jaccard, avg_coverage = evaluate_step(model, valid_data_loader)
            print(
                "Epoch {}, Steps {}, Train loss {:.6}, Average Jaccard {:.4}, Average coverage {:.4}".format(
                    epoch, step_count, train_loss, avg_jaccard, avg_coverage
                )
            )
        else:
            print("Epoch {}, Steps {}, Train loss {:.6}".format(epoch, step_count, train_loss))

        # TODO rarely ever saves the model
        if epoch % 100 == 0:
            torch.save(model.state_dict(), config.model_save_path + "transformer_" + str(epoch))


if __name__ == "__main__":
    main()
