"""
Bigram model
"""
import torch
import matplotlib

matplotlib.use("qtagg")
import matplotlib.pyplot as plt

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

print(f"Original name length:{len(names)}")

# Calculate the lengths of all names
num_names = len(names)
name_lengths = [len(name) for name in names]
max_name_length = max(name_lengths)
min_name_length = min(name_lengths)
print(f"Number of names: {num_names}")
print(f"Max name length: {max_name_length}")
print(f"Min name length: {min_name_length}")
print(f"Average name length: {sum(name_lengths) / num_names}")

# Create a histogram of name lengths
plt.hist(name_lengths, bins=range(1, 17), align='left', rwidth=0.8)
plt.xlabel('Name Length')
plt.xticks(range(0, 17))
plt.ylabel('Frequency')
plt.title('Distribution of Name Lengths')
# plt.show()

# filter out names 10 characters and longer
names = [name for name in names if len(name) < 10]
# Add special character boundary to names
print(f"Shortened names: {len(names)}")

# Bigram distribution of characters of 27 x 27 matrix
# Each row represents the probability of a character following another character
# P("aa"), P("ab"), ... fpr the given names
bigram = torch.zeros((vocab_size, vocab_size))
total = 0
for name in names:
    # '$olivia$'
    # [('$', 'o'), ('o', 'l'), ('l', 'i'), ('i', 'v'), ('v', 'i'), ('i', 'a'), ('a', '$')]
    for ch1, ch2 in zip(name, name[1:]):
        ch1_int = ch_to_i[ch1]
        ch2_int = ch_to_i[ch2]
        bigram[ch1_int][ch2_int] += 1
        total += 1
bigram /= total

# This is sort of better
# jon, zis, khrikyasahvi, sakanee, ana
# The problem with this is
# 1. Probability increase exponential with vocabulary size
# 2. Based on previous character. With more previous characters, exponential increase
for i in range(5):
    generated = "$"
    while True:
        bigram_probs = bigram[ch_to_i[generated[-1]]]
        sampled_char = i_to_ch[
            torch.multinomial(bigram_probs, 1).item()
        ]
        if sampled_char == "$":
            break
        generated += sampled_char
    print(f"name {i}: {generated[1:]}")
