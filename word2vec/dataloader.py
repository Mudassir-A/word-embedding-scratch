import re
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def clean_and_tokenize(text):
    cleaned_text = re.sub(r"[^a-zA-Z]", " ", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    cleaned_text = cleaned_text.lower()
    tokens = cleaned_text.split(" ")
    with open("../data/stopwords-en.txt", "r") as f:
        stop_words = f.read()
    stop_words = stop_words.replace("\n", " ").split(" ")
    return [token for token in tokens if token not in stop_words[:-1]]


def target_context_tuples(tokens, window_size):
    context = []
    for i, token in enumerate(tokens):
        context_words = [t for t in merge(tokens, i, window_size) if t != token]
        for c in context_words:
            context.append((token, c))
    return context


def merge(tokens, i, window_size):
    left_id = i - window_size if i >= window_size else i - 1 if i != 0 else i
    right_id = i + window_size + 1 if i + window_size <= len(tokens) else len(tokens)
    return tokens[left_id:right_id]


class W2VDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        context = self.df["context_ohe"][index]
        target = self.df["target_ohe"][index]
        return context, target


def prepare_data(data_path, window_size=2, batch_size=64):
    with open(data_path, "r") as f:
        data = f.read()

    tokens = clean_and_tokenize(data)

    unique_words = set(tokens)
    word_i = {word: i for (i, word) in enumerate(unique_words)}
    i_word = {i: word for (i, word) in enumerate(unique_words)}

    target_context_pairs = target_context_tuples(tokens, window_size)
    df = pd.DataFrame(target_context_pairs, columns=["target", "context"])

    vocab_size = len(unique_words)
    token_indexes = [word_i[token] for token in unique_words]
    encodings = F.one_hot(torch.tensor(token_indexes), num_classes=vocab_size).float()

    df["target_ohe"] = df["target"].apply(lambda x: encodings[word_i[x]])
    df["context_ohe"] = df["context"].apply(lambda x: encodings[word_i[x]])

    dataset = W2VDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, word_i, i_word, vocab_size
