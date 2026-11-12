"""
This is MLP model which is better than bigram model.
It trains embeddings and predicts the next character similar to bigram.
However, it scales linearly with the length of vocabulary.
"""
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import plotly.graph_objects as go

vocab = "$abcdefghijklmnopqrstuvwxyz"
vocab_size = len(vocab)
# Create dictionaries for character to index and index to character
ch_to_i = {char: i for i, char in enumerate(vocab)}
i_to_ch = {i: char for i, char in enumerate(vocab)}

names = []
with open('data/names_2022.txt', 'r') as file:
    for line in file:
        name, _, _ = line.lower().strip().split(',')
        names.append("$" + name + "$")

# This is an example of how to encode and decode a string
# example_name = "$ada$"
# Define encode and decode functions
# encode = [0, 1, 4, 1, 0]
encode = lambda word: torch.tensor([ch_to_i[c] for c in word])
decode = lambda tensor_i: ''.join(i_to_ch[i.item()] for i in tensor_i)
# print(f"Encode: {encode(example_name)}")
# print(f"Decode: {decode(encode(example_name))}")

# These a global objects...
name_indices = [encode(name) for name in names]
# At each position, the label is the next character.
# name_indices   (X):   $  a  d  a  $      → [0, 1, 4, 1, 0]
# target_indices (Y):      a  d  a  $      → [1, 4, 1, 0]
target_indices = [name_index[1:] for name_index in name_indices]
# pad_sequences - pad 0 up the max
X = pad_sequence(name_indices, batch_first=True, padding_value=0)
# trick to pad Y with a tensor for size 11, then remove the last row
max_name_length = max(len(name) for name in names)
target_indices.append(torch.empty((max_name_length), dtype=torch.long))
Y = pad_sequence(target_indices, batch_first=True, padding_value=-1)[:-1]


def show_plot():
    # Assume embeddings is your (28, 3) tensor
    embeddings = model.embedding.weight.detach()
    # Create a list of labels
    labels = ['$', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
              'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '*']
    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        z=embeddings[:, 2],
        mode='markers+text',
        text=labels,  # add labels
        textfont=dict(size=12, color="white"),
        marker=dict(size=8),
        textposition='middle center',  # position the labels
        name="characters"
    )])
    # Add x, y, and z axes as lines
    fig.add_trace(go.Scatter3d(x=[-2, 2], y=[0, 0], z=[0, 0], mode='lines',
                               line=dict(color='black', dash='solid', width=4), name='x-axis'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-3, 2], z=[0, 0], mode='lines',
                               line=dict(color='black', dash='dashdot', width=4), name='y-axis'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-2, 2], mode='lines',
                               line=dict(color='black', dash='longdash', width=4), name='z-axis'))
    fig.update_layout(
        autosize=False,
        width=800,
        height=700,
        legend=dict(
            orientation="v",
            xanchor="center",
            yanchor="bottom",
            x=.9,
            y=.5,
            font_size=12,
            bordercolor="black",
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )
    # Show the plot
    fig.show()


def get_batch(batch_size=64):
    """
    Helper function to create a batch of input-target pairs.
    Used to create the training data.
    """
    random_idx = torch.randint(0, X.size(0), (batch_size,))
    inputs = X[random_idx]
    labels = Y[random_idx]
    return inputs, labels


# Example embeddings
# embedding_dim = 3
# embedding = nn.Embedding(vocab_size, embedding_dim)
# example_input = torch.tensor([1, 1, 0, 2])
# input_embd = embedding(example_input)
# print(input_embd.shape)
# print(input_embd)
# inputs, labels = get_batch(3)
# print(inputs)
# print(labels)

class SequenceMLP(nn.Module):
    def __init__(self, vocab_size, max_sequence_length, embedding_dim, hidden_dim=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 27 x 3
        self.linear = nn.Linear(embedding_dim * max_sequence_length, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        sequence_embeddings = torch.zeros(batch_size, seq_len, self.max_sequence_length * self.embedding_dim)
        for i in range(seq_len):  # 0..16. 17 is the max sequence length in the current batch
            subsequence = torch.zeros(batch_size, self.max_sequence_length, dtype=torch.int)
            prefix = x[:, :i + 1]
            subsequence[:, :i + 1] = prefix
            emb = self.embedding(subsequence)
            sequence_embeddings[:, i, :] = emb.view(batch_size, -1)  # flatten into a list of values
        x = self.linear(sequence_embeddings)  # Passed through a hidden layer
        x = self.relu(x)
        x = self.out(x)
        return x


embedding_dim = 3
max_sequence_length = X.shape[1]
model = SequenceMLP(vocab_size, max_sequence_length, embedding_dim)


def train(model, optimizer, num_steps=10_001, loss_report_interval=1_000):
    losses = []
    for i in range(1, num_steps):
        inputs, labels = get_batch()
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-1)
        losses.append(loss.item())
        if i % loss_report_interval == 0:
            print(f'Average loss at step {i}: {sum(losses[-loss_report_interval:]) / loss_report_interval:.4f}')
        loss.backward()
        optimizer.step()


optimizer = optim.SGD(model.parameters(), lr=0.1)
train(model, optimizer)

show_plot()


def generate_samples(model, num_samples=1, max_len=max_name_length):
    sequences = torch.zeros((num_samples, 1)).int()
    for _ in range(max_len):
        logits = model(sequences)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        sequences = torch.cat((sequences, idx_next), dim=1)

    for sequence in sequences:
        indices = torch.where(sequence == 0)[0]
        end = indices[1] if len(indices) > 1 else max_len
        sequence = sequence[1:end]
        print(decode(sequence))


"""
vahalleye
neyolal
elevoi
sede
luscide
iracnan
"""
generate_samples(model, num_samples=10)
