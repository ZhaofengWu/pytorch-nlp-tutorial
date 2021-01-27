"""
Sentiment classification example using GRU on the imdb dataset.
Note that in practice it is horrible style to put everything in one file.
You probably want to use a GPU to run this file, as it can take many hours per epoch on CPUs,
unless you are willing to use extreme hyperparameters.

For brevity, we omit certain practices that are usually done in real systems. For example:
(1) data persistence (e.g., with pickle) so we don't have to download and process every time;
(2) dropout;
(3) learning rate scheduling (see torch.optim.lr_scheduler);
(4) model saving/loading;
(5) early stopping;
(6) choosing the best epoch based on development set performance;
(7) hyperparameter tuning.

[Here](https://github.com/pytorch/examples) are some more official examples provided by pytorch.
"""

from collections import Counter
import os
import random
import tarfile
import tempfile
import urllib.request

import nltk
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Punctuation info for tokenization
nltk.download('punkt')

# Special tokens. It is important that they don't appear in the actual vocab, hence this weird look.
# Sometimes you want bos and eos too.
PAD = "@@PAD@@"
UNK = "@@UNK@@"

# Hyperparameters. This configuration has been tested on a 11GB GPU. You can change them if you
# have a smaller GPU, in which case we recommend starting with MAX_SEQ_LEN and UNK_THRESHOLD.
MAX_SEQ_LEN = -1  # -1 for no truncation
UNK_THRESHOLD = 5
BATCH_SIZE = 128
N_EPOCHS = 20
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
N_RNN_LAYERS = 2


def seed_everything(seed=1):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def download_data():
    """
    A function to download and uncompress the imdb data. You don't have to understand anything here.
    """

    def extract_data(dir, split):
        data = []
        for label in ("pos", "neg"):
            label_dir = os.path.join(dir, "aclImdb", split, label)
            files = sorted(os.listdir(label_dir))
            for file in files:
                filepath = os.path.join(label_dir, file)
                with open(filepath, encoding="UTF-8") as f:
                    data.append({"raw": f.read(), "label": label})
        return data

    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    stream = urllib.request.urlopen(url)
    tar = tarfile.open(fileobj=stream, mode="r|gz")
    with tempfile.TemporaryDirectory() as td:
        tar.extractall(path=td)
        train_data = extract_data(td, "train")
        test_data = extract_data(td, "test")
        return train_data, test_data


def split_data(train_data, num_split=2000):
    """Splits the training data into training and development sets."""
    random.shuffle(train_data)
    return train_data[:-num_split], train_data[-num_split:]


def tokenize(data, max_seq_len=MAX_SEQ_LEN):
    """
    Here we use nltk to tokenize data. There are many othe possibilities. We also truncate the
    sequences so that the training time and memory is more manageable. You can think of truncation
    as making a decision only looking at the first X words.
    """
    for example in data:
        example["text"] = []
        for sent in nltk.sent_tokenize(example["raw"]):
            example["text"].extend(nltk.word_tokenize(sent))
        if max_seq_len >= 0:
            example["text"] = example["text"][:max_seq_len]


def create_vocab(data, unk_threshold=UNK_THRESHOLD):
    """
    Creates a vocabulary with tokens that have frequency above unk_threshold and assigns each token
    a unique index, including the special tokens.
    """
    counter = Counter(token for example in data for token in example["text"])
    vocab = {token for token in counter if counter[token] > unk_threshold}
    print(f"Vocab size: {len(vocab) + 2}")  # add the special tokens
    print(f"Most common tokens: {counter.most_common(10)}")
    token_to_idx = {PAD: 0, UNK: 1}
    for token in vocab:
        token_to_idx[token] = len(token_to_idx)
    return token_to_idx


def apply_vocab(data, token_to_idx):
    """
    Applies the vocabulary to the data and maps the tokenized sentences to vocab indices as the
    model input.
    """
    for example in data:
        example["text"] = [token_to_idx.get(token, token_to_idx[UNK]) for token in example["text"]]


def apply_label_map(data, label_to_idx):
    """Converts string labels to indices."""
    for example in data:
        example["label"] = label_to_idx[example["label"]]


class SentimentDataset(Dataset):
    """
    Sometimes it suffices to use the pytorch built-in TensorDataset, but here we want to control the
    padding on a finer-grained level, so we implement our own. Specifically, we batch together
    similar-lengthed examples to minimize padding and speed up training. Most heavy-lifting is done
    in collate_fn that pads the tensors and batches them together.
    """

    def __init__(self, data, pad_idx):
        data = sorted(data, key=lambda example: len(example["text"]))
        self.texts = [example["text"] for example in data]
        self.labels = [example["label"] for example in data]
        self.pad_idx = pad_idx

    def __getitem__(self, index):
        return [self.texts[index], self.labels[index]]

    def __len__(self):
        return len(self.texts)

    def collate_fn(self, batch):
        def tensorize(elements, dtype):
            return [torch.tensor(element, dtype=dtype) for element in elements]

        def pad(tensors):
            """Assumes 1-d tensors."""
            max_len = max(len(tensor) for tensor in tensors)
            padded_tensors = [
                F.pad(tensor, (0, max_len - len(tensor)), value=self.pad_idx) for tensor in tensors
            ]
            return padded_tensors

        texts, labels = zip(*batch)
        return [
            torch.stack(pad(tensorize(texts, torch.long)), dim=0),
            torch.stack(tensorize(labels, torch.long), dim=0),
        ]


class SequenceClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_labels, n_rnn_layers, pad_idx):
        super().__init__()

        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(
            embedding_dim, hidden_dim, num_layers=n_rnn_layers, batch_first=True, bidirectional=True
        )
        # We take the final hidden state at all GRU layers as the sequence representation.
        # 2 because bidirectional.
        layered_hidden_dim = hidden_dim * n_rnn_layers * 2
        self.output = nn.Linear(layered_hidden_dim, n_labels)

    def forward(self, text):
        # text shape: (batch_size, max_seq_len) where max_seq_len is the max length *in this batch*
        # lens shape: (batch_size,)
        non_padded_positions = text != self.pad_idx
        lens = non_padded_positions.sum(dim=1)

        # embedded shape: (batch_size, max_seq_len, embedding_dim)
        embedded = self.embedding(text)
        # You can pass the embeddings directly to the RNN, but as the input potentially has
        # different lengths, how do you know when to stop unrolling the recurrence for each example?
        # pytorch provides a util function pack_padded_sequence that converts padded sequences with
        # potentially different lengths into a special PackedSequence object that keeps track of
        # these things. When passing a PackedSequence object into the RNN, the output will be a
        # PackedSequence too (but not the hidden state as that always has a length of 1). Since we
        # do not use the per-token output, we do not unpack it. But if you need it, e.g. for
        # token-level classification such as POS tagging, you can use pad_packed_sequence to convert
        # it back to a regular tensor.
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        # nn.GRU produces two outputs: one is the per-token output and the other is per-sequence.
        # The pers-sequence output is simiar to the last per-token output, except that it is taken
        # at all layers.
        # output (after unpacking) shape: (batch_size, max_seq_len, hidden_dim)
        # hidden shape: (n_layers * n_directions, batch_size, hidden_dim)
        packed_output, hidden = self.rnn(packed_embedded)
        # shape: (batch_size, n_layers * n_directions * hidden_dim)
        hidden = hidden.transpose(0, 1).reshape(hidden.shape[1], -1)
        # Here we directly output the raw scores without softmax normalization which would produce
        # a valid probability distribution. This is because:
        # (1) during training, pytorch provides a loss function "F.cross_entropy" that combines
        # "softmax + F.nll_loss" in one step. See the `train` function below.
        # (2) during evaluation, we usually only care about the class with the highest score, but
        # not the actual probablity distribution.
        # shape: (batch_size, n_labels)
        return self.output(hidden)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, dataloader, optimizer, device):
    for texts, labels in tqdm(dataloader):
        texts, labels = texts.to(device), labels.to(device)
        output = model(texts)
        loss = F.cross_entropy(output, labels)
        model.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, dataloader, device):
    count = correct = 0.0
    with torch.no_grad():
        for texts, labels in tqdm(dataloader):
            texts, labels = texts.to(device), labels.to(device)
            # shape: (batch_size, n_labels)
            output = model(texts)
            # shape: (batch_size,)
            predicted = output.argmax(dim=-1)
            count += len(predicted)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {correct / count}")


def main():
    seed_everything()

    print("Downloading data")
    train_data, test_data = download_data()
    train_data, dev_data = split_data(train_data)
    print(f"Data sample: {train_data[:3]}")
    print(f"train {len(train_data)}, dev {len(dev_data)}, test {len(test_data)}")

    print("Processing data")
    for data in (train_data, dev_data, test_data):
        tokenize(data)
    # Here we only use the training data to create the vocabulary because
    # (1) we shouldn't look at the test set; and
    # (2) we want the dev set to accurately reflect the test set performance.
    # There are people who do other things.
    token_to_idx = create_vocab(train_data)
    label_to_idx = {"neg": 0, "pos": 1}
    for data in (train_data, dev_data, test_data):
        apply_vocab(data, token_to_idx)
        apply_label_map(data, label_to_idx)

    pad_idx = token_to_idx[PAD]
    train_dataset = SentimentDataset(train_data, pad_idx)
    dev_dataset = SentimentDataset(dev_data, pad_idx)
    test_dataset = SentimentDataset(test_data, pad_idx)
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, collate_fn=dev_dataset.collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, collate_fn=test_dataset.collate_fn
    )

    model = SequenceClassifier(
        len(token_to_idx), EMBEDDING_DIM, HIDDEN_DIM, len(label_to_idx), N_RNN_LAYERS, pad_idx
    )
    print(f"Model has {count_parameters(model)} parameters.")
    # Adam is just a fancier version of SGD.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Random baseline")
    evaluate(model, dev_dataloader, device)
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch + 1}")  # 0-based -> 1-based
        train(model, train_dataloader, optimizer, device)
        evaluate(model, dev_dataloader, device)
    print(f"Test set performance")
    evaluate(model, test_dataloader, device)


if __name__ == "__main__":
    main()
