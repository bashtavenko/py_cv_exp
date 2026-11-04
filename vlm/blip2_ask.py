"""
BLIP-2: caption an image and ask questions about it (VQA).

Example output:
  Q: What is in this image?   A: the patriots are in a 3-4 defense
  Q: What is 3-4 defense?     A: a defensive alignment used to defend the run and pass
"""
import torch
from PIL import Image
from absl import app, flags, logging
from transformers import Blip2ForConditionalGeneration, Blip2Processor

FLAGS = flags.FLAGS

flags.DEFINE_string("image", "images/all-22.jpeg", "Path to an image")
flags.DEFINE_list(
    "questions",
    ["What is in this image?", "What is 3-4 defense?"],
    "Comma-separated questions to ask about the image",
)
flags.DEFINE_string("model_id", "Salesforce/blip2-opt-2.7b", "BLIP-2 checkpoint on the Hugging Face Hub")
flags.DEFINE_integer("max_new_tokens", 30, "Max tokens to generate per answer")


def run(_argv):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    logging.info("Loading %s on %s (first run downloads several GB)", FLAGS.model_id, device)

    processor = Blip2Processor.from_pretrained(FLAGS.model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        FLAGS.model_id, dtype=dtype
    ).to(device).eval()

    image = Image.open(FLAGS.image).convert("RGB")

    def generate(prompt: str | None = None) -> str:
        # noinspection PyArgumentList
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = inputs.to(device=device, dtype=dtype)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=FLAGS.max_new_tokens)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    logging.info("Caption: %s", generate())

    for q in FLAGS.questions:
        logging.info("Q: %s | A: %s", q, generate(f"Question: {q} Answer:"))


if __name__ == "__main__":
    app.run(run)