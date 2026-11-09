from pathlib import Path
import torch

import dlcv2026
import linear_classifier

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

print("PyTorch:", torch.__version__)
print("Implementation:", Path(linear_classifier.__file__).resolve())


X_demo = torch.tensor([[1.0, 2.0], [-1.0, 1.0]])
W_demo = torch.tensor([[0.5, -1.0, 0.0], [1.0, 0.5, -0.5]])
b_demo = torch.tensor([0.1, 0.0, 0.2])

scores_demo = linear_classifier.linear_scores(X_demo, W_demo, b_demo)
predictions_demo = scores_demo.argmax(dim=1)
print("scores:\n", scores_demo)
print("predicted classes:", predictions_demo.tolist())

assert scores_demo.shape == (2, 3)
assert predictions_demo.dtype == torch.int64


X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
W = torch.tensor([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
y = torch.tensor([2, 0, 1], dtype=torch.int64)

loss, gradient = linear_classifier.svm_loss_naive(
    W.clone(), X, y, reg=0.0
)
print("unregularized loss:", float(loss))
print("gradient:\n", gradient)

assert loss.ndim == 0
assert gradient.shape == W.shape
assert torch.allclose(loss, torch.tensor(2.0), atol=1e-12)


help(dlcv2026.grad_check_sparse)

torch.manual_seed(4)
X_check = torch.randn(7, 5)
W_check = 0.1 * torch.randn(5, 4)
y_check = torch.tensor([0, 1, 2, 3, 0, 1, 2], dtype=torch.int64)

loss_check, grad_check = linear_classifier.svm_loss_naive(
    W_check, X_check, y_check, reg=0.0
)
f = lambda weights: linear_classifier.svm_loss_naive(
    weights, X_check, y_check, reg=0.0
)[0]
errors = dlcv2026.grad_check_sparse(
    f, W_check, grad_check, num_checks=10, h=1e-6
)
print("maximum relative error:", max(errors))
assert max(errors) < 1e-5