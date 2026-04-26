import torch
from torch.utils.data import Dataset
from collections import Counter


class EncodingsDataset(Dataset):
    def __init__(self, dataframe, vocab=None, label_map=None, max_len=20):
        self.data = dataframe
        self.max_len = max_len
        self.sequences = [list(str(text)) for text in self.data['encoded']]
        
        if vocab is None:
            all_chars = [char for seq in self.sequences for char in seq]
            char_counts = Counter(all_chars)
            self.vocab = {char: i+2 for i, (char, _) in enumerate(char_counts.most_common(100))}
            self.vocab['<PAD>'] = 0
            self.vocab['<UNK>'] = 1
        else:
            self.vocab = vocab

        if label_map is None:
            unique_labels = sorted(self.data['encoding'].unique())
            self.label_map = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Text to Indices
        seq = self.sequences[idx]
        idx_seq = [self.vocab.get(char, 1) for char in seq]
        
        # Padding
        if len(idx_seq) < self.max_len:
            idx_seq += [0] * (self.max_len - len(idx_seq))
        else:
            idx_seq = idx_seq[:self.max_len]
            
        # Label to Integer
        label_str = self.data.iloc[idx]['encoding']
        label_idx = self.label_map[label_str]
            
        return torch.tensor(idx_seq), torch.tensor(label_idx)