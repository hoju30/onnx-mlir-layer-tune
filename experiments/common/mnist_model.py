"""Shared MNIST demo-net forward pass (MaxPool2d -> Reshape -> Linear(fc1) ->
ReLU -> Linear(fc2) -> Softmax) and calibration-image loading, used by both
build_posit_graph.py (quantization-aware activation stats) and
calibrate_activations.py (standalone calibration/verification tool).

Kept in its own module, rather than defined in either of those two files, so
neither has to import from the other (build_posit_graph.py already gets
imported BY calibrate_activations.py for decode_all_constants -- a reverse
import would be circular).
"""
import numpy as np


def maxpool2x2(x):
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def softmax(x, axis=1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def forward(images, fc1_w, fc1_b, fc2_w, fc2_b):
    """Returns a dict of the 6 canonical nodes' output activations."""
    maxpool_out = maxpool2x2(images)
    reshape_out = maxpool_out.reshape(len(images), -1)
    gemm3_out = reshape_out @ fc1_w.T + fc1_b
    relu_out = np.maximum(gemm3_out, 0)
    gemm5_out = relu_out @ fc2_w.T + fc2_b
    softmax_out = softmax(gemm5_out)
    return {
        "MaxPool_0": maxpool_out, "Reshape_2": reshape_out, "Gemm_3": gemm3_out,
        "Relu_4": relu_out, "Gemm_5": gemm5_out, "Softmax_6": softmax_out,
    }


def load_mnist(n, train):
    import torchvision, torchvision.transforms as T
    ds = torchvision.datasets.MNIST(
        "/tmp/mnist_data", train=train, download=True,
        transform=T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))]))
    idx = np.linspace(0, len(ds) - 1, n).astype(int) if n < len(ds) else np.arange(len(ds))
    imgs = np.stack([np.array(ds[i][0]) for i in idx])
    lbls = np.array([ds[i][1] for i in idx])
    return imgs.astype(np.float32), lbls


def value_stats(arr, prefix):
    """Shared mean/std/p99_abs/zero_ratio/dynamic_range stats -- used for
    both weight_* (claude.md 8.2) and activation_* (8.3) node features."""
    flat = np.asarray(arr, dtype=np.float64).ravel()
    return {
        f"{prefix}_mean": float(flat.mean()),
        f"{prefix}_std": float(flat.std()),
        f"{prefix}_p99_abs": float(np.percentile(np.abs(flat), 99)),
        f"{prefix}_zero_ratio": float((flat == 0).mean()),
        f"{prefix}_dynamic_range": float(flat.max() - flat.min()),
    }
