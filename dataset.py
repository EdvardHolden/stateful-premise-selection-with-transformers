import numpy as np
from torch.utils.data import Dataset
from vocabulary import SourceVocabulary, TargetVocabulary

# class Statement2PremisesDataset(Dataset):
#
#     def __init__(self, source_path, target_path, source_vocab, target_vocab):
#         self.source_vocab = SourceVocabulary()
#         self.target_vocab = TargetVocabulary()
#         with open(source_path, 'r') as source_file, open(target_path, 'r') as target_file:
#             source_lines = []
#             target_lines = []
#             while True:
#                 source_line = source_file.readline().strip('\n')
#                 target_line = target_file.readline().strip('\n')
#                 if ((not source_line) or (not target_line)):
#                     break
#                 source_lines.append(source_line)
#                 target_lines.append(target_line)
#                 self.source_vocab.add_sentence(source_line)
#                 self.target_vocab.add_sentence(target_line)
#         self.statement_data = [ self.source_vocab.sentence2indices(line) for line in source_lines ]
#         self.premises_data = [ self.target_vocab.sentence2indices(line) for line in target_lines ]
#
#     def __len__(self):
#         return len(self.statement_data)
#
#     def __getitem__(self, item):
#         statement_indices = self.statement_data[item]
#         premises_indices = self.premises_data[item]
#         return { 'source': np.asarray(statement_indices, dtype = np.int64),
#                  'source_lengths': len(statement_indices),
#                  'target': np.asarray(premises_indices, dtype = np.int64),
#                  'target_lengths': len(premises_indices) }


def get_tokenised_data(file_path, vocab, max_length=None):
    with open(file_path, "r") as file:
        source_lines = []
        while True:
            source_line = file.readline().strip("\n").split()
            if not source_line: # Check whether we are finished
                break
            if max_length is not None: # Truncate if set
                source_line = source_line[:max_length]
            source_lines.append(source_line)
    return [vocab.sentence2indices(line) for line in source_lines]



class StatementDataset(Dataset):
    def __init__(self, source_path, source_vocab, conjecture_max_length):

        self.statement_data = get_tokenised_data(source_path, source_vocab, max_length=conjecture_max_length)

    def __len__(self):
        return len(self.statement_data)

    def __getitem__(self, item):
        statement_indices = self.statement_data[item]
        return {
            "source": np.asarray(statement_indices, dtype=np.int64),
            "source_lengths": len(statement_indices),
        }


class Statement2PremisesDataset(Dataset):
    def __init__(self, source_path, target_path, source_vocab, target_vocab, conjecture_max_length):

        self.statement_data = get_tokenised_data(source_path, source_vocab, max_length=conjecture_max_length)
        self.premises_data = get_tokenised_data(target_path, target_vocab)

    def __len__(self):
        return len(self.statement_data)

    def __getitem__(self, item):
        statement_indices = self.statement_data[item]
        premises_indices = self.premises_data[item]
        return {
            "source": np.asarray(statement_indices, dtype=np.int64),
            "source_lengths": len(statement_indices),
            "target": np.asarray(premises_indices, dtype=np.int64),
            "target_lengths": len(premises_indices),
        }
