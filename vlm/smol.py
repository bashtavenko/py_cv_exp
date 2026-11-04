"""
SmolVLM2: describe a video and ask questions about the action in it.

Example (tennis clip):
  What is happening in this video?  A: The video captures a tennis match in progress on a clay court ...
  Who wins the point?               A: The player in the white shirt and blue shorts wins ...

Run:
  python smol.py --video videos/tennis.mp4
"""
import torch
from absl import app, flags, logging
from transformers import AutoModelForImageTextToText, AutoProcessor

FLAGS = flags.FLAGS

flags.DEFINE_string("video", "videos/tennis.mp4", "Path to a video file")
flags.DEFINE_list(
    "questions",
    [
        "Describe what is happening in this video.",
        "Who wins the point?",
    ],
    "Comma-separated prompts to ask about the video",
)
flags.DEFINE_string(
    "model_id",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "Any SmolVLM2 checkpoint, e.g. HuggingFaceTB/SmolVLM2-500M-Video-Instruct "
    "(smaller/faster) or HuggingFaceTB/SmolVLM2-256M-Video-Instruct (tiniest)",
)
flags.DEFINE_integer("max_new_tokens", 128, "Max tokens to generate per answer")


def run(_argv):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    logging.info("Loading %s on %s (first run downloads a few GB)", FLAGS.model_id, device)

    processor = AutoProcessor.from_pretrained(FLAGS.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        FLAGS.model_id, torch_dtype=dtype
    ).to(device).eval()

    def ask(prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": FLAGS.video},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device, dtype=dtype)

        with torch.no_grad():
            ids = model.generate(**inputs, do_sample=False, max_new_tokens=FLAGS.max_new_tokens)

        # The output contains the prompt too
        new_ids = ids[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    for q in FLAGS.questions:
        logging.info("Q: %s\nA: %s", q, ask(q))


if __name__ == "__main__":
    app.run(run)