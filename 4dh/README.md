# 4D-Humans / HRM2.0

4D-Humans no longer works with latest PyTorch and Conda won't help there.

The main problem is `detectron2`. It wants the latest PyTorch, but for example on my machine I have not the latest CUDA 12.8, but torch wants 13.0.
`pip install torch` brings CUDA 13.0

```
pip install -r requirements-torch.txt
pip install --no-build-isolation -r requirements-4dh.txt
pip install -r requirements-4dh-extra.txt
```