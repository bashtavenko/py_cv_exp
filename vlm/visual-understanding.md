# Visual Understanding

I have a first version of Richard Szeliski's Computer Vision: Algorithms and Applications (2010).
On page 3 he writes: "...the dream of having a computer interpret an image at the same level as a two-year old (say
counting
all of the animals in a picture) remains elusive".

I think there are three separate questions here:

1. Has computer vision achieved the ability to recognize and describe what is in an image?
2. Has it achieved the ability to understand events, actions, relationships and temporal evolution?
3. Has it developed a sufficiently grounded model of the world to reason reliably about what is happening and what will
   happen next?
   The answer to the first is largely yes. The second is partially. The third remains a substantial open problem.

## What happened to computer vision after 2010?

The classical approach for the first question has largely been achieved. For example, to understand an image containing
a person riding a bicycle, a system might:

1. Detect regions and edges.
2. Identify objects.
3. Segment the person and bicycle.
4. Estimate their geometry and relative positions.
5. Recognize the human pose.
6. Infer the interaction between person and bicycle.
   Each component had a relatively explicit purpose.

Deep learning changed this. Rather than manually deciding which features should represent a bicycle, a person, or a
particular interaction, neural networks learned representations from large datasets.

The famous AlexNet breakthrough was one important turning point. Subsequently, detection, segmentation, pose
estimation, action recognition and other tasks were increasingly dominated by learned architectures.

Then the next big turning point was the Transformer architecture that lead to the Vison Language Models (VLA) and
derivatives of Video Language Models, and Vision-Language-Action.

## Are these models essentially next-token prediction machines?

Technically yes, many modern VLMs are autoregressive generative models, but it does not mean that next-token prediction
does not necessarily mean that the model has learned only superficial statistical associations.

To predict accurately across sufficiently diverse situations, a neural network may develop internal representations of
objects, spatial arrangements, relationships and regularities in the physical world.

However, the existence of useful internal representations does not establish that the model has learned a robust,
causally grounded world model.

In particular, a model might recognize that a cup is falling without accurately representing its trajectory, velocity,
collision, or eventual resting position. Recognizing the concept of falling is not equivalent to understanding falling.
A two-year-old child has concepts of gravity, shapes and motion. The child action is grounded on reality.

For example, a child learns that:

* Objects continue to exist when hidden behind other objects.
* Objects cannot normally pass through solid surfaces.
* A falling object will eventually hit something.
* Pushing an object can change its position.
* The same object can be viewed from different perspectives.
* An action has consequences.

This knowledge is acquired through continuous interaction with the physical environment, not simply by studying pictures
accompanied by descriptions.

## What does current research tells us?

### Video understanding is often less temporal than it appears.

Apple's October 2025 [research](https://machinelearning.apple.com/research/breaking-down?utm_source=chatgpt.com) on
video-language benchmarks found that models can sometimes answer questions without actually watching the video. They
also found that certain questions could be answered equally well after shuffling the video frames. In other words,
models could exploit language priors or static visual information instead of understanding the sequence of events

### More demanding video reasoning remains difficult

The [GLIMPSE benchmark](https://aclanthology.org/2025.emnlp-main.1415/?utm_source=chatgpt.com), published at EMNLP 2025,
specifically tests video understanding that requires examining the full video.

It contains 3,269 videos and more than 4,300 questions. Human participants achieved 94.82% accuracy, while the
best-performing model reported in the paper achieved 66.43%.

The benchmark investigates trajectory analysis, temporal reasoning and other tasks that cannot be solved simply by
looking at a few representative frames.

### Physical and causal reasoning are still significant weaknesses.

Meta's [CausalVQA benchmark](https://ai.meta.com/research/publications/causalvqa-a-physically-grounded-causal-reasoning-benchmark-for-video-models/?utm_source=chatgpt.com)
investigates whether models understand what might happen after an action, including hypothetical situations,
anticipation, planning and counterfactual reasoning.

Its researchers found that frontier multimodal models remained substantially below human performance, especially in
anticipation and hypothetical reasoning.