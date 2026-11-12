"""
This is the same embeddings model but with self-attention mechanism.
Self-attention lets each position to look back and decide how much to
read from each, based on content.

q = x Wq,  k = x Wk,  v = x Wv
weights = softmax(q kᵀ / √d)    # how relevant is each earlier token?
out = weights @ v               # weighted mix of their information

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

vocab = "$" + ''.join(chr(ord('a') + i) for i in range(26))
vocab_size = len(vocab)
# Create dictionaries for character to index and index to character
ch_to_i = {char: i for i, char in enumerate(vocab)}
i_to_ch = {i: char for i, char in enumerate(vocab)}

# Define encode and decode functions
encode = lambda word: torch.tensor([ch_to_i[c] for c in word])
decode = lambda tensor_i: ''.join(i_to_ch[i.item()] for i in tensor_i)

# Get data
names = []
with open('data/names_2022.txt', 'r') as file:
    for line in file:
        name, _, _ = line.lower().strip().split(',')
        names.append(name)

# filter out names 10 characters and longer
names = [name for name in names if len(name) < 10]
# Add special character boundary to names
names = ['$' + name + '$' for name in names]

# Length of max name
max_name_length = max(len(name) for name in names)
longest_name = max(names, key=len)
longest_name_index = names.index(longest_name)

# Create a function to get a batch of data
names_index = [torch.tensor([ch_to_i[char] for char in name]) for name in names]
targets_index = [name_index[1:] for name_index in names_index]
names_index[0], targets_index[0]

from torch.nn.utils.rnn import pad_sequence

X = pad_sequence(names_index, batch_first=True, padding_value=0)
# trick to pad Y with -1 and the same size as X
targets_index.append(X[0])
Y = pad_sequence(targets_index, batch_first=True, padding_value=-1)[:-1]


def get_batch(batch_size=64):
    random_idx = torch.randint(0, X.size(0), (batch_size,))
    batch = X[random_idx]
    labels = Y[random_idx]
    return batch, labels


batch, labels = get_batch()


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
Query, Key, Value concept playground
"""
torch.manual_seed(0)
x = torch.rand(4, 3)

# Liner transformation v_m = x_m * W_m^T  + b_m
query = F.linear(x, weight=torch.rand(3, 3), bias=torch.rand(3))
key = F.linear(x, weight=torch.rand(3, 3), bias=torch.rand(3))
value = F.linear(x, weight=torch.rand(3, 3), bias=torch.rand(3))


def dot_product_attention_single(q, k, v):
    attn_weights = q @ k.T  # DOT product over N inputs
    attn_weights = F.softmax(attn_weights, dim=-1)  # Get attention weights
    output = attn_weights @ v  # DOT product
    return output


dot_product_attention_single(query, key, value)

"""
Scaled dot product.  Causal attention playground.
"""
torch.manual_seed(0)
x = torch.rand(1, 4, 3)  # batch size 1, 3D tensor with 1 x 4 x 3

query = F.linear(x, weight=torch.rand(3, 3), bias=torch.rand(3))
key = F.linear(x, weight=torch.rand(3, 3), bias=torch.rand(3))
value = F.linear(x, weight=torch.rand(3, 3), bias=torch.rand(3))


def scaled_dot_product_causal_attention(q, k, v):
    # assumes batch dimension is present
    # Transpose to swap the last two dimensions
    # First dimension remains the batch dimension
    attn_weights = q @ k.transpose(1, 2)
    # create a mask to prevent the model from attending to future tokens
    # Lower triangular matrix of 1s and then fill the upper
    # triangular matrix with -inf
    mask = torch.tril(torch.ones(attn_weights.shape[1:]), diagonal=0)
    attn_weights = attn_weights.masked_fill(mask == 0, value=float('-inf'))
    # Normalize the attention weights. Prevents gradient explosion.
    attn_weights = attn_weights / torch.sqrt(torch.tensor(k.shape[-1]).float())
    attn_weights = F.softmax(attn_weights, dim=-1)
    output = attn_weights @ v
    return output, attn_weights


output, attn_weights = scaled_dot_product_causal_attention(query, key, value)
# Pytorch already implements this
expected_output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
print(torch.allclose(output, expected_output))


class AttentionMLP(nn.Module):
    def __init__(self, n_embd, vocab_size, block_size, n_hidden=64):
        super().__init__()
        self.tok_embd = nn.Embedding(vocab_size, n_embd)
        self.attn_weights = None

        self.query_proj = nn.Linear(n_embd, n_embd)
        self.key_proj = nn.Linear(n_embd, n_embd)
        self.value_proj = nn.Linear(n_embd, n_embd)

        self.register_buffer("mask", torch.tril(torch.ones((block_size, block_size)), diagonal=0))

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_embd)
        )

        self.output_proj = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        x = self.tok_embd(x)
        batch_size, seq_len, embd_dim = x.shape

        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        attn_weights = q @ k.transpose(1, 2)
        attn_weights = attn_weights.masked_fill(self.mask[:seq_len, :seq_len] == 0, value=float('-inf'))
        attn_weights = attn_weights / torch.sqrt(torch.tensor(k.shape[-1]).float())
        self.attn_weights = F.softmax(attn_weights, dim=-1)
        x = self.attn_weights @ v
        x = self.mlp(x)

        x = self.output_proj(x)
        return x


model = AttentionMLP(32, vocab_size, max_name_length)
optimizer = optim.SGD(model.parameters(), lr=0.01)
train(model, optimizer, num_steps=10_001, loss_report_interval=1_000)

# arey
# iain
# keesa
# nebcanne
# digshonli
generate_samples(model, 10)


def visualize_attention(attention_weights, name):
    plt.figure(figsize=(30, 15))
    ax = sns.heatmap(attention_weights.cpu().detach().numpy(), annot=True, cmap="YlGnBu", annot_kws={"size": 25})
    labels = list(name)
    ticks = [i + 0.5 for i in range(len(labels))]
    plt.xticks(ticks=ticks, labels=labels, rotation='horizontal')
    plt.yticks(ticks=ticks, labels=labels, rotation='horizontal')
    plt.tick_params(axis='both', which='major', labelsize=25)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)
    plt.show()


# Row 3, 'o' position. Primary focus on 'o' itself and considers
# 0.11 for 'j'
name = "$" + "john" + "$"
encoded_name = encode(name)
model(encoded_name.unsqueeze(0))
attn_weights = model.attn_weights[0]
visualize_attention(attn_weights, name)
