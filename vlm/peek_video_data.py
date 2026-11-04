"""
Peek at video-language training data: what does a (video, prompt, answer) sample look like?

Uses FineVideo (HuggingFaceFV/finevideo), one of the video datasets in SmolVLM2's training mix.
Each sample is an mp4 plus rich JSON metadata with timecodes (scenes, activities), which is
exactly the temporal signal a video model learns from. We stream the dataset, keep only
"Sports" clips, and for a few of them:
  * save the mp4
  * save a contact sheet of evenly spaced frames (the "image" side of the pair)
  * print text pairs built from the metadata (the "prompt -> answer" side)

FineVideo is GATED: open https://huggingface.co/datasets/HuggingFaceFV/finevideo, accept the
terms, then run `huggingface-cli login` (or set HF_TOKEN).

Install:
  pip install -U datasets pillow opencv-python absl-py

Run:
  python peek_video_data.py
  python peek_video_data.py --keyword=tennis --num_videos=3
  python peek_video_data.py --category "" --keyword=football   # search all categories

Streaming downloads video bytes while scanning, so --max_scan keeps it bounded. A narrow
keyword may need a bigger --max_scan.

Examples
Sample 3: Sports / Documentary Profiles | 30s
  video:  peek_out/sample_3.mp4
  frames: peek_out/sample_3_frames.png

  PROMPT: Describe what happens in this video, in order.
  ANSWER:
  [00:00:00.000 - 00:00:10.844] Omarion is being interviewed and speaks about fatherhood.
  [00:00:11.011 - 00:00:12.304] The two girls hug and smile brightly.
  [00:00:14.014 - 00:00:19.019] A still photo of a younger Omarion is shown, with his hands clasped together in prayer.
  [00:00:21.021 - 00:00:22.856] A still photo shows Omarion and another male both dressed in white, posing together.

  PROMPT: What happens between 00:00:00.000 and 00:00:10.844?
  ANSWER: Omarion is being interviewed and speaks about fatherhood.

  PROMPT: What happens between 00:00:11.011 and 00:00:12.304?
  ANSWER: The two girls hug and smile brightly.

  PROMPT: What object is Omarion wearing in his left ear?
  ANSWER: He is wearing a gold ankh earring.

  PROMPT: What color shirts are Omarion and the other male wearing in the still photo?
  ANSWER: They are both wearing white shirts.
"""
import json
import os

import cv2
from PIL import Image, ImageDraw
from absl import app, flags, logging
from datasets import load_dataset

FLAGS = flags.FLAGS

flags.DEFINE_string("category", "Sports", "content_parent_category to keep ('' = any)")
flags.DEFINE_string("keyword", "", "Only keep clips whose metadata mentions this word, e.g. tennis")
flags.DEFINE_integer("num_videos", 3, "How many videos to show (1-5 is plenty)")
flags.DEFINE_float("max_seconds", 60.0, "Skip videos longer than this")
flags.DEFINE_integer("max_scan", 500, "Max dataset rows to scan while looking for matches")
flags.DEFINE_integer("num_frames", 8, "Frames per contact sheet")
flags.DEFINE_string("out", "peek_out", "Output directory")


def find_scenes(meta):
    """FineVideo keeps timecoded scenes in the JSON; be tolerant about exactly where."""
    for holder in (meta, meta.get("content_metadata") or {}):
        scenes = holder.get("scenes")
        if scenes:
            return scenes
    return []


def activity_rows(meta):
    """Flatten scenes -> (start, end, description) tuples."""
    rows = []
    for scene in find_scenes(meta):
        for act in scene.get("activities") or []:
            ts = act.get("timestamp") or act.get("timestamps") or {}
            rows.append((
                ts.get("start_timestamp", "?"),
                ts.get("end_timestamp", "?"),
                act.get("description", "").strip(),
            ))
    return [r for r in rows if r[2]]


def matches(meta, keyword):
    if not keyword:
        return True
    return keyword.lower() in json.dumps(meta).lower()


def contact_sheet(video_path, n, out_path):
    """Grab n evenly spaced frames, label each with its timestamp, tile into one image."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tiles = []
    for i in range(n):
        idx = int((i + 0.5) * total / n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.thumbnail((320, 240))
        ImageDraw.Draw(img).text((6, 6), f"t={idx / fps:.1f}s", fill=(255, 255, 0))
        tiles.append(img)
    cap.release()
    if not tiles:
        return False
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    w, h = tiles[0].size
    sheet = Image.new("RGB", (cols * w, rows * h))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, (i // cols) * h))
    sheet.save(out_path)
    return True


def show_pairs(meta):
    """Print the (prompt -> answer) pairs we can build from this sample's metadata."""
    acts = activity_rows(meta)
    pairs = []
    if acts:
        story = " ".join(f"[{s} - {e}] {d}" for s, e, d in acts)
        pairs.append(("Describe what happens in this video, in order.", story))
        for s, e, d in acts[:2]:
            pairs.append((f"What happens between {s} and {e}?", d))
    qa = (meta.get("content_metadata") or {}).get("qAndA") or meta.get("qAndA") or []
    for item in qa[:2]:
        if isinstance(item, dict) and item.get("question") and item.get("answer"):
            pairs.append((item["question"], item["answer"]))
    if not pairs:
        print("  (no activities/Q&A found; top-level keys:", list(meta.keys()), ")")
    for q, a in pairs:
        print(f"  PROMPT: {q}\n  ANSWER: {a[:400]}{'...' if len(a) > 400 else ''}\n")


def run(_argv):
    os.makedirs(FLAGS.out, exist_ok=True)
    ds = load_dataset("HuggingFaceFV/finevideo", split="train", streaming=True)

    found = 0
    for scanned, sample in enumerate(ds, start=1):
        if scanned > FLAGS.max_scan:
            logging.info("Scanned %d rows without enough matches; raise --max_scan.", FLAGS.max_scan)
            break
        meta = sample["json"]
        if FLAGS.category and meta.get("content_parent_category") != FLAGS.category:
            continue
        if meta.get("duration_seconds", 0) > FLAGS.max_seconds or not matches(meta, FLAGS.keyword):
            continue

        found += 1
        video_path = os.path.join(FLAGS.out, f"sample_{found}.mp4")
        with open(video_path, "wb") as f:
            f.write(sample["mp4"])
        sheet_path = os.path.join(FLAGS.out, f"sample_{found}_frames.png")
        contact_sheet(video_path, FLAGS.num_frames, sheet_path)

        print("=" * 80)
        print(f"Sample {found}: {meta.get('content_parent_category')} / {meta.get('content_fine_category')}"
              f" | {meta.get('duration_seconds')}s")
        print(f"  video:  {video_path}\n  frames: {sheet_path}\n")
        show_pairs(meta)

        if found >= FLAGS.num_videos:
            break

    if found == 0:
        logging.warning("No matching samples found. Try --category '' or a larger --max_scan.")


if __name__ == "__main__":
    app.run(run)