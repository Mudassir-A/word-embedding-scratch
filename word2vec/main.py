import torch
import torch.nn.functional as F

from model import Word2Vec
from dataloader import prepare_data
from training import train_word2vec


def main():
    DATA_PATH = "../data/data.txt"
    WINDOW_SIZE = 2
    BATCH_SIZE = 64
    EMBED_SIZE = 10
    NUM_EPOCHS = 300
    LEARNING_RATE = 1e-2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader, word_i, i_word, vocab_size = prepare_data(
        DATA_PATH, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE
    )

    model = Word2Vec(vocab_size, EMBED_SIZE).to(DEVICE)
    model, loss_history = train_word2vec(
        model,
        dataloader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
    )

    def find_similar_words(word, top_k=5):
        word_encoding = F.one_hot(
            torch.tensor(word_i[word]), num_classes=vocab_size
        ).float()
        pred = model(word_encoding.to(DEVICE))
        similar_indices = torch.argsort(pred, descending=True).squeeze(0)[:top_k]
        return [i_word[i.item()] for i in similar_indices]

    print("Similar words to 'language':", find_similar_words("language"))
    print("Similar words to 'life':", find_similar_words("life"))
    print("Embedding for 'biology':", model.get_word_embedding(word_i["biology"]))


if __name__ == "__main__":
    main()
