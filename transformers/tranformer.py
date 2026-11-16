"""
Transformer
Has residual connections (skip connections to deal with vanishing gradients),
multi-head attention, positional embeddings, layer normalization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

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

X = pad_sequence(names_index, batch_first=True, padding_value=0)
# trick to pad Y with -1 and the same size as X
targets_index.append(X[0])
Y = pad_sequence(targets_index, batch_first=True, padding_value=-1)[:-1]


def get_batch(batch_size=64):
    random_idx = torch.randint(0, X.size(0), (batch_size,))
    batch = X[random_idx]
    labels = Y[random_idx]
    return batch, labels

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


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads=4, n_hidden=64):
        super().__init__()
        assert n_embd % num_heads == 0, "Embedding dimension must be divisible by the number of heads"

        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads

        self.query_proj = nn.Linear(n_embd, n_embd)
        self.key_proj = nn.Linear(n_embd, n_embd)
        self.value_proj = nn.Linear(n_embd, n_embd)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_embd)
        )

        self.norm_1 = nn.LayerNorm(n_embd)
        self.norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape

        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        # multiheaded attention
        q = q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        # attention
        attn_weights = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # multiple heads concatenation
        attn_weights = attn_weights.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)

        # norm and residual connections here
        x = self.norm_1(x + attn_weights)
        x = self.norm_2(x + self.mlp(x))
        return x


class Transformer(nn.Module):
    def __init__(self, n_embd, vocab_size, block_size, num_blocks=6):
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, n_embd)
        # Without positional embeddings, the transformer would treat
        # input seq as unordered set of tokens.
        self.positional_embedding = nn.Embedding(block_size, n_embd)

        # Multihead attention
        # Each head has its own set of learnable parameters (queries, keys, values)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(n_embd) for _ in range(num_blocks)]
        )

        self.output_proj = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        _, seq_len = x.shape

        pos_embd = self.positional_embedding(torch.arange(seq_len))  # <1>
        char_embd = self.char_embedding(x)
        x = char_embd + pos_embd
        x = self.transformer_blocks(x)
        x = self.output_proj(x)

        return x


n_embd = 64
model = Transformer(n_embd, vocab_size, block_size=max_name_length)
optimizer = optim.SGD(model.parameters(), lr=0.1)
train(model, optimizer, num_steps=4_001, loss_report_interval=1_000)

"""
estemalia
praffin
adriel
feory
cayde
"""
generate_samples(model, num_samples=10)
