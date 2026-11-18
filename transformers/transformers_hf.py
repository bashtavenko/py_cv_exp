"""
Name generation with Hugging Face: tokenizers + transformers (+ pipeline).
Can be done with GPT-2, Llama, etc.
Courtesy of Claude
"""
from tokenizers import Tokenizer, Regex, models, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing
from transformers import (PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, pipeline)
import string

# 1. Data: lowercase ascii names shorter than 10 characters
names = [line.split(",")[0].lower() for line in open("data/names_2022.txt")]
names = [n for n in names if n.isalpha() and n.isascii() and len(n) < 10]

# 2. Tokenizer: one token per character, every name wrapped as <bos> ... <eos>
vocab = {t: i for i, t in
         enumerate(["<pad>", "<bos>", "<eos>"] + list(string.ascii_lowercase))}
tok = Tokenizer(models.WordLevel(vocab))
tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")
tok.post_processor = TemplateProcessing(single="<bos> $A <eos>",
                                        special_tokens=[("<bos>", 1),
                                                        ("<eos>", 2)])
tok.decoder = decoders.Fuse()
tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<bos>",
                                    eos_token="<eos>", pad_token="<pad>")

# 3. Dataset: a plain list of dicts is enough for Trainer
data = [{"input_ids": ids} for ids in tokenizer(names).input_ids]

# 4. Model: a tiny GPT-2 (embeddings + positions + 4 attention blocks)
model = GPT2LMHeadModel(
    GPT2Config(vocab_size=len(vocab), n_positions=16, n_embd=64,
               n_layer=4, n_head=4, bos_token_id=1, eos_token_id=2,
               pad_token_id=0))

# 5. Train: the collator pads batches and builds the labels (next-token
# prediction)
Trainer(model=model,
        args=TrainingArguments("name-gpt", per_device_train_batch_size=64,
                               num_train_epochs=10, learning_rate=2e-3,
                               save_strategy="no", report_to="none"),
        train_dataset=data,
        data_collator=DataCollatorForLanguageModeling(tokenizer,
                                                      mlm=False)).train()

# 6. Generate: start from <bos>, sample until the model emits <eos>
generate = pipeline("text-generation", model=model, tokenizer=tokenizer)
for out in generate("<bos>", max_new_tokens=10, do_sample=True, top_k=0,
                    num_return_sequences=10):
    print(out["generated_text"])
