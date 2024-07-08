import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

from transformers import BertTokenizer

from models import NaiveBaseline


def load_torch_dataset() -> dict[str, Dataset]:
    # Load hf dataset
    datasets = load_dataset('imdb')

    datasets['train'] = datasets['train'].select(range(100))
    datasets['test'] = datasets['test'].select(range(100))
    datasets['unsupervised'] = datasets['unsupervised'].select(range(100))

    # tokenize dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def tokenize_function(samples):
        return tokenizer(
            samples['text'],
            padding='max_length',
            truncation=True
        )
    tokenized_datasets = datasets.map(tokenize_function, batched=True)

    # Set format to PyTorch (with train and test)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    return tokenized_datasets

def load_train_test_dataloader(datasets: dict[str, Dataset]) -> tuple[DataLoader, DataLoader]:
    num_samples = 100
    return DataLoader(
        dataset=datasets['train'].select(range(num_samples)),
        batch_size=8,
        shuffle=True
    ), DataLoader(
        dataset=datasets['test'].select(range(num_samples)),
        batch_size=8
    )

def train(dataloader: DataLoader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    # training loop
    for batch, X in enumerate(dataloader):
        X_id = X['input_ids'].to(device)

        # Compute prediction error: error at predicting inputs in this case
        pred = model(X_id)
        loss = loss_fn(pred, X_id)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader: DataLoader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_test_loop():
    # identify available devices
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    
    # load Dataloaders
    datasets = load_torch_dataset()
    train_data_loader, test_data_loader = load_train_test_dataloader(datasets)

    # model init and train mode
    model = NaiveBaseline().to(device)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # loss
    loss_fn = nn.CrossEntropyLoss()

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_data_loader, model, loss_fn, optimizer, device)
        test(test_data_loader, model, loss_fn, device)
    print("Done!")

if __name__ == "__main__":
    train_test_loop()