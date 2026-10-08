"""
BLIP-2: caption an image and ask questions about it (VQA).

What is in this image? Answer: The patriots are in a 3-4 defense
What is 3-4 defense? 3-4 defense is a defensive alignment that is used to defend the run and pass

What is this image? It's a picture of a football player.
What is the numer of his jersey? 21
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

MODEL_ID = "Salesforce/blip2-opt-2.7b"


def run(_argv):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    logging.info("Loading %s on %s (first run downloads several GB)", MODEL_ID, device)

    processor = Blip2Processor.from_pretrained(MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype
    ).to(device).eval()

    image = Image.open(FLAGS.image).convert("RGB")

    def generate(prompt=None):
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=30)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    logging.info("Caption: %s", generate())

    for q in FLAGS.questions:
        logging.info("Q: %s | A: %s", q, generate(f"Question: {q} Answer:"))


if __name__ == "__main__":
    app.run(run)
