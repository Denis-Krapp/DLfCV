from pathlib import Path
import importlib
import sys
import time

import matplotlib.pyplot as plt
import torch

candidate_dirs = [
    Path.cwd(),
    Path.cwd() / "Computer Vision" / "Lab05",
    Path.cwd() / "Lab05",
]
lab_dir = next(
    (
        path.resolve()
        for path in candidate_dirs
        if (path / "linear_classifier.py").is_file()
        and (path / "dlcv2026").is_dir()
    ),
    None,
)
if lab_dir is None:
    raise FileNotFoundError(
        "Could not find the Lab05 support files. Keep linear_classifier.py "
        "and the dlcv2026 folder together with the notebook."
    )
if str(lab_dir) not in sys.path:
    sys.path.insert(0, str(lab_dir))

import dlcv2026
import linear_classifier

importlib.reload(linear_classifier)

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

print("PyTorch:", torch.__version__)
print("Lab directory:", lab_dir)
print("Implementation:", Path(linear_classifier.__file__).resolve())


importlib.reload(linear_classifier)

X_fixture = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
W_fixture = torch.tensor([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
y_fixture = torch.tensor([2, 0, 1], dtype=torch.int64)

data_loss, data_gradient = linear_classifier.svm_loss_naive(
    W_fixture.clone(), X_fixture, y_fixture, reg=0.0
)
regularized_loss, regularized_gradient = linear_classifier.svm_loss_naive(
    W_fixture.clone(), X_fixture, y_fixture, reg=0.1
)

print("data loss:", float(data_loss))
print("regularized loss:", float(regularized_loss))
assert torch.allclose(data_loss, torch.tensor(2.0), atol=1e-12)
assert torch.allclose(regularized_loss, torch.tensor(2.4), atol=1e-12)
assert torch.allclose(
    regularized_gradient - data_gradient,
    2 * 0.1 * W_fixture,
    atol=1e-12,
)


importlib.reload(linear_classifier)

torch.manual_seed(6)
X_check = torch.randn(7, 5)
W_check = 0.1 * torch.randn(5, 4)
y_check = torch.tensor([0, 1, 2, 3, 0, 1, 2], dtype=torch.int64)

_, grad_check = linear_classifier.svm_loss_naive(
    W_check, X_check, y_check, reg=0.03
)
f = lambda weights: linear_classifier.svm_loss_naive(
    weights, X_check, y_check, reg=0.03
)[0]
errors = dlcv2026.grad_check_sparse(
    f, W_check, grad_check, num_checks=10, h=1e-6
)
print("maximum relative error:", max(errors))
assert max(errors) < 1e-5


importlib.reload(linear_classifier)

torch.manual_seed(7)
X_small = torch.randn(8, 5)
W_small = 0.05 * torch.randn(5, 4)
y_small = torch.tensor([0, 1, 2, 3, 0, 2, 1, 3], dtype=torch.int64)

loss_naive, grad_naive = linear_classifier.svm_loss_naive(
    W_small, X_small, y_small, reg=0.05
)
loss_vectorized, grad_vectorized = linear_classifier.svm_loss_vectorized(
    W_small, X_small, y_small, reg=0.05
)

print("loss difference:", float((loss_naive - loss_vectorized).abs()))
print("gradient max difference:", float((grad_naive - grad_vectorized).abs().max()))
assert torch.allclose(loss_naive, loss_vectorized, atol=1e-10)
assert torch.allclose(grad_naive, grad_vectorized, atol=1e-10)


importlib.reload(linear_classifier)

torch.manual_seed(8)
X_speed = torch.randn(512, 128)
W_speed = 0.01 * torch.randn(128, 10)
y_speed = torch.randint(0, 10, (512,))

start = time.perf_counter()
linear_classifier.svm_loss_naive(W_speed, X_speed, y_speed, reg=1e-3)
naive_seconds = time.perf_counter() - start

start = time.perf_counter()
linear_classifier.svm_loss_vectorized(
    W_speed, X_speed, y_speed, reg=1e-3
)
vectorized_seconds = time.perf_counter() - start

print(f"naive: {naive_seconds:.4f}s")
print(f"vectorized: {vectorized_seconds:.4f}s")
print(f"speedup: {naive_seconds / max(vectorized_seconds, 1e-12):.1f}x")



importlib.reload(linear_classifier)

X_index = torch.arange(30, dtype=torch.float64).reshape(10, 3)
y_index = torch.arange(10, dtype=torch.int64)
generator = torch.Generator().manual_seed(3)
X_batch, y_batch = linear_classifier.sample_batch(
    X_index, y_index, 6, generator
)

print("batch labels:", y_batch.tolist())
assert X_batch.shape == (6, 3)
assert y_batch.shape == (6,)
assert torch.equal(X_batch[:, 0] / 3, y_batch.to(torch.float64))



data = dlcv2026.make_toy_classification(seed=11)
for name, value in data.items():
    print(name, tuple(value.shape))

plt.scatter(
    data["X_train"][:, 0],
    data["X_train"][:, 1],
    c=data["y_train"],
    cmap="viridis",
    s=16,
)
plt.title("Training data")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.show()



importlib.reload(linear_classifier)

W_trained, loss_history = linear_classifier.train_linear_classifier(
    linear_classifier.svm_loss_vectorized,
    data["X_train"],
    data["y_train"],
    learning_rate=0.08,
    reg=1e-3,
    num_iters=400,
    batch_size=64,
    seed=5,
)

plt.plot(loss_history)
plt.xlabel("iteration")
plt.ylabel("minibatch loss")
plt.title("SVM training history")
plt.show()

train_predictions = linear_classifier.predict_linear_classifier(
    W_trained, data["X_train"]
)
val_predictions = linear_classifier.predict_linear_classifier(
    W_trained, data["X_val"]
)
train_accuracy = linear_classifier.accuracy(
    train_predictions, data["y_train"]
)
val_accuracy = linear_classifier.accuracy(
    val_predictions, data["y_val"]
)
print(f"training accuracy: {100 * train_accuracy:.2f}%")
print(f"validation accuracy: {100 * val_accuracy:.2f}%")

assert loss_history[-1] < loss_history[0]
assert val_accuracy > 0.85



importlib.reload(linear_classifier)

W_strong_reg, strong_reg_history = (
    linear_classifier.train_linear_classifier(
        linear_classifier.svm_loss_vectorized,
        data["X_train"],
        data["y_train"],
        learning_rate=0.08,
        reg=1.0,
        num_iters=400,
        batch_size=64,
        seed=5,
    )
)

strong_reg_predictions = linear_classifier.predict_linear_classifier(
    W_strong_reg, data["X_val"]
)
strong_reg_accuracy = linear_classifier.accuracy(
    strong_reg_predictions, data["y_val"]
)
baseline_norm = torch.linalg.vector_norm(W_trained).item()
strong_reg_norm = torch.linalg.vector_norm(W_strong_reg).item()

print(f"baseline validation accuracy: {100 * val_accuracy:.2f}%")
print(f"strong-reg validation accuracy: {100 * strong_reg_accuracy:.2f}%")
print(f"baseline weight norm: {baseline_norm:.4f}")
print(f"strong-reg weight norm: {strong_reg_norm:.4f}")

assert strong_reg_norm < baseline_norm



