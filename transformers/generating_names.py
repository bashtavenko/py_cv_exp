"""
Return names with equal probability which completely horrible.
"""
import torch
import torch.nn.functional as F

# Uses '$' as a new line character
vocab = "$abcdefghijklmnopqrstuvwxyz"
vocab_size = len(vocab)
# Create dictionaries for character to index and index to character
ch_to_i = {char: i for i, char in enumerate(vocab)}
i_to_ch = {i: char for i, char in enumerate(vocab)}

equal_probs = F.softmax(torch.ones(vocab_size), dim=0)  # 0.0370 which is 1/27
for i in range(5):
    generated = ""
    while True:
        # Returns a single index sampled from equal_probs distribution
        random_int = torch.multinomial(equal_probs, 1).item()
        random_char = i_to_ch[random_int]
        if random_char == "$":
            break
        generated += random_char
    # Horrible
    print(f"name {i}: {generated}")
