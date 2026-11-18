"""
Tokenization example. Generating text is the same as generating a list of names.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

"""
Basics 
"""
# Create a word-level tokenizer
tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
# Use whitespace pre-tokenizer
tokenizer.pre_tokenizer = Whitespace()
# Define the trainer
trainer = WordLevelTrainer(special_tokens=["[UNK]"])
tokenizer.train(["data/odyssey.txt"], trainer=trainer)
vocab_size = tokenizer.get_vocab_size()
print(f"\nSize: {vocab_size}")
decode = tokenizer.decode
encode = tokenizer.encode

"""
Encode text example
"""
with open("data/odyssey.txt", "r", encoding="utf-8") as f:
    text = f.read()
# Tokenize the text file
encoding = tokenizer.encode(text)
# Create sequences of 100 tokens
sequence_length = 100
X, Y = [], []
for i in range(0, len(encoding.ids) - sequence_length, sequence_length):
    X.append(encoding.ids[i:i + sequence_length])
    Y.append(encoding.ids[i + 1:i + sequence_length + 1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.tensor(X).to(device)
Y = torch.tensor(Y).to(device)
print(X.shape)


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
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]),
                               labels.view(-1), ignore_index=-1)
        losses.append(loss.item())
        if i % loss_report_interval == 0:
            print(
                f"Average loss at step {i}: "
                f"{sum(losses[-loss_report_interval:]) / loss_report_interval:.4f}")
        loss.backward()
        optimizer.step()


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads=4, n_hidden=64):
        super().__init__()
        assert n_embd % num_heads == 0, ("Embedding dimension must be "
                                         "divisible by the number of heads")

        self.num_heads = num_heads  # <1>
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
        q = q.view(batch_size, sequence_length, self.num_heads,
                   self.head_dim).transpose(1, 2)  # <1>
        k = k.view(batch_size, sequence_length, self.num_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.num_heads,
                   self.head_dim).transpose(1, 2)

        # attention
        attn_weights = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # multiple heads concatenation
        attn_weights = attn_weights.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, -1)

        # norm and residual connections here
        x = self.norm_1(x + attn_weights)
        x = self.norm_2(x + self.mlp(x))
        return x


class Transformer(nn.Module):
    def __init__(self, n_embd, vocab_size, block_size, num_blocks=6):
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding = nn.Embedding(block_size, n_embd)

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(n_embd) for _ in range(num_blocks)]
        )

        self.output_proj = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        _, seq_len = x.shape

        pos_embd = self.positional_embedding(
            torch.arange(seq_len).to(device))  # <1>
        char_embd = self.char_embedding(x)
        x = char_embd + pos_embd
        x = self.transformer_blocks(x)
        x = self.output_proj(x)

        return x


def generate_samples(model, num_samples=1, max_len=sequence_length):
    sequences = torch.zeros((num_samples, 1)).int().to(device)
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
        decoded_sequence = decode(sequence.tolist())
        print(format_sequence(decoded_sequence))


def format_sequence(sequence):
    # This function formats the sequence to handle punctuation and spacing
    # correctly.
    formatted_sequence = ""
    for i, char in enumerate(sequence):
        if char in ",.;:!?":
            formatted_sequence = formatted_sequence.rstrip() + char + " "
        else:
            formatted_sequence += char
    return formatted_sequence.strip()


n_embd = 64
model = Transformer(n_embd, vocab_size, block_size=sequence_length)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)
train(model, optimizer, num_steps=501, loss_report_interval=100)

generate_samples(model)
