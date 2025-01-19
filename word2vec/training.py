from torch import nn
import torch.optim as optim


def train_word2vec(
    model, dataloader, num_epochs=300, learning_rate=1e-2, device="cuda"
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)
    loss_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch, (context, target) in enumerate(dataloader):
            context = context.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(context)
            loss = criterion(pred, target)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch+1} | Loss: {epoch_loss}")

        loss_history.append(epoch_loss)

    return model, loss_history
