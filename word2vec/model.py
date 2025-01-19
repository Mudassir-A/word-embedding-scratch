import torch.nn as nn

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.linear_1 = nn.Linear(vocab_size, embed_size)
        self.linear_2 = nn.Linear(embed_size, vocab_size, bias=False)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x

    def get_word_embedding(self, word_i):
        embeddings = self.linear_2.weight.detach().cpu()
        return embeddings[word_i]