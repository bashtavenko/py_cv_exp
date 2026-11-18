"""
Downloads CIFAR-10 and ImageNet datasets.
"""
import os

from absl import app
from absl import flags
from absl import logging
from datasets import load_dataset
from huggingface_hub import login

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    'dataset',
    'cifar10',
    ['cifar10', 'imagenet-1k'],
    'Dataset to fetch: "cifar10" or "imagenet-1k".',
)
flags.DEFINE_string(
    'split',
    'train',
    'Dataset split (e.g., "train", "test", "validation").',
)
flags.DEFINE_integer(
    'num_images',
    5,
    'Number of sample images to extract and save.',
)
flags.DEFINE_string(
    'hf_token',
    None,
    'Hugging Face access token (required for gated datasets like imagenet-1k).',
)
flags.DEFINE_string(
    'output_dir',
    'data/images',
    'Directory where extracted PNG images will be saved.',
)

DATASET_MAP = {
    'cifar10': 'uoft-cs/cifar10',
    'imagenet-1k': 'ILSVRC/imagenet-1k',
}


def main(argv):
    del argv  # Unused.
    logging.set_verbosity(logging.INFO)

    # Authenticate with Hugging Face if a token is provided (or cached via CLI)
    if FLAGS.hf_token:
        logging.info('Logging into Hugging Face Hub...')
        login(token=FLAGS.hf_token)

    repo_id = DATASET_MAP[FLAGS.dataset]
    logging.info(
        'Streaming "%s" (%s) split="%s"...',
        FLAGS.dataset,
        repo_id,
        FLAGS.split,
    )

    try:
        # streaming=True avoids downloading the entire dataset upfront
        ds = load_dataset(repo_id, split=FLAGS.split, streaming=True)
    except Exception as e:
        logging.error('Failed to load "%s": %s', repo_id, e)
        if FLAGS.dataset == 'imagenet-1k':
            logging.info(
                'May need to accepted the terms at '
                'https://huggingface.co/datasets/ILSVRC/imagenet-1k and '
                'have a valid --hf_token.'
            )
        return

    # CIFAR-10 uses 'img' as the column name; ImageNet uses 'image'
    img_key = 'img' if FLAGS.dataset == 'cifar10' else 'image'
    label_names = ds.features['label'].names

    save_path = os.path.join(FLAGS.output_dir, FLAGS.dataset, FLAGS.split)
    os.makedirs(save_path, exist_ok=True)

    for idx, sample in enumerate(ds.take(FLAGS.num_images)):
        image = sample[img_key]
        label_idx = sample['label']

        # ImageNet labels can contain commas/slashes (e.g., "tench,
        # Tinca tinca")
        raw_label = label_names[label_idx].split(',')[0]
        clean_label = raw_label.replace('/', '_').replace(' ', '_')

        filepath = os.path.join(save_path, f'{idx:04d}_{clean_label}.png')
        image.save(filepath)
        logging.info('Saved [%d/%d]: %s', idx + 1, FLAGS.num_images, filepath)

    logging.info('Done! Exported %d images to %s.', FLAGS.num_images, save_path)


if __name__ == '__main__':
    app.run(main)
