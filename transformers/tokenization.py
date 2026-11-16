"""
Tokenization example. This could work the same way to generate text
"""
import torch

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
Encode text
"""
with open('data/odyssey.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# Tokenize the text file
encoding = tokenizer.encode(text)
# Create sequences of 100 tokens
sequence_length = 100
X, Y = [], []
for i in range(0, len(encoding.ids) - sequence_length, sequence_length):
    X.append(encoding.ids[i:i+sequence_length])
    Y.append(encoding.ids[i+1:i+sequence_length+1])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X = torch.tensor(X).to(device)
Y = torch.tensor(Y).to(device)
print(X.shape)
