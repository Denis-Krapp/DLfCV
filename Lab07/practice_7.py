from pathlib import Path
import importlib
import sys

import matplotlib.pyplot as plt
import torch

candidate_dirs = [
    Path.cwd(),
    Path.cwd() / "Computer Vision" / "Lab07",
    Path.cwd() / "Lab07",
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
        "Could not find the Lab07 support files. Keep linear_classifier.py "
        "and the dlcv2026 folder together with the notebook."
    )

if str(lab_dir) not in sys.path:
    sys.path.insert(0, str(lab_dir))
for module_name in tuple(sys.modules):
    if (
        module_name == "linear_classifier"
        or module_name == "dlcv2026"
        or module_name.startswith("dlcv2026.")
    ):
        del sys.modules[module_name]

import dlcv2026
import linear_classifier

importlib.reload(linear_classifier)

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

print("PyTorch:", torch.__version__)
print("Lab directory:", lab_dir)
print("Implementation:", Path(linear_classifier.__file__).resolve())


importlib.reload(linear_classifier)

W_example = torch.tensor([1.0, -2.0])
gradient_example = torch.tensor([0.5, -0.25])
W_sgd, _ = linear_classifier.sgd_update(
    W_example,
    gradient_example,
    {"learning_rate": 0.1},
)
print("SGD result:", W_sgd)
assert torch.allclose(W_sgd, torch.tensor([0.95, -1.975]))



importlib.reload(linear_classifier)

config = {"learning_rate": 0.1, "momentum": 0.9}
W_first, config = linear_classifier.sgd_momentum_update(
    W_example, gradient_example, config
)
W_second, config = linear_classifier.sgd_momentum_update(
    W_first, gradient_example, config
)

print("after one step:", W_first)
print("after two steps:", W_second)
print("velocity:", config["velocity"])
assert torch.allclose(W_first, torch.tensor([0.95, -1.975]))
assert torch.allclose(W_second, torch.tensor([0.855, -1.9275]))
assert torch.allclose(
    config["velocity"], torch.tensor([-0.095, 0.0475])
)


def quadratic_gradient(position):
    return torch.stack((position[0], 80.0 * position[1]))


def optimizer_path(update_rule, config, steps=60):
    position = torch.tensor([-4.0, 3.0])
    path = [position.clone()]
    for _ in range(steps):
        position, config = update_rule(
            position, quadratic_gradient(position), config
        )
        path.append(position.clone())
    return torch.stack(path)


sgd_path = optimizer_path(
    linear_classifier.sgd_update,
    {"learning_rate": 0.018},
)
momentum_path = optimizer_path(
    linear_classifier.sgd_momentum_update,
    {"learning_rate": 0.018, "momentum": 0.8},
)

x_grid = torch.linspace(-4.5, 4.5, 240)
y_grid = torch.linspace(-3.5, 3.5, 240)
grid_x, grid_y = torch.meshgrid(x_grid, y_grid, indexing="xy")
objective = 0.5 * (grid_x**2 + 80.0 * grid_y**2)

plt.contour(
    grid_x,
    grid_y,
    objective,
    levels=torch.logspace(-1, 3, 16),
)
plt.plot(sgd_path[:, 0], sgd_path[:, 1], "o-", ms=2, label="SGD")
plt.plot(
    momentum_path[:, 0],
    momentum_path[:, 1],
    "o-",
    ms=2,
    label="momentum",
)
plt.xlabel("parameter 1")
plt.ylabel("parameter 2")
plt.title("Optimization paths")
plt.legend()
plt.show()


W_adagrad, adagrad_config = linear_classifier.adagrad_update(
    W_example,
    gradient_example,
    {"learning_rate": 0.1},
)
W_rmsprop, rmsprop_config = linear_classifier.rmsprop_update(
    W_example,
    gradient_example,
    {"learning_rate": 0.1, "decay_rate": 0.9},
)
W_nesterov, nesterov_config = linear_classifier.nesterov_update(
    W_example,
    gradient_example,
    {"learning_rate": 0.1, "momentum": 0.9},
)

print("Nesterov result:", W_nesterov)
print("AdaGrad result:", W_adagrad)
print("RMSProp result:", W_rmsprop)
assert (adagrad_config["squared_gradient_sum"] > 0).all()
assert (rmsprop_config["squared_gradient_average"] > 0).all()
assert "velocity" in nesterov_config


importlib.reload(linear_classifier)

adam_config = {
    "learning_rate": 0.01,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
}
W_adam_first, adam_config = linear_classifier.adam_update(
    W_example, gradient_example, adam_config
)
W_adam_second, adam_config = linear_classifier.adam_update(
    W_adam_first, gradient_example, adam_config
)

print("after one step:", W_adam_first)
print("after two steps:", W_adam_second)
print("timestep:", adam_config["timestep"])
assert torch.allclose(
    W_adam_first,
    torch.tensor([0.99, -1.99]),
    atol=1e-8,
)
assert torch.allclose(
    W_adam_second,
    torch.tensor([0.98, -1.98]),
    atol=1e-8,
)
assert adam_config["timestep"] == 2


importlib.reload(linear_classifier)
data = dlcv2026.make_toy_classification(seed=11)

experiments = {
    "SGD": (
        linear_classifier.sgd_update,
        {"learning_rate": 0.08},
    ),
    "Momentum": (
        linear_classifier.sgd_momentum_update,
        {"learning_rate": 0.03, "momentum": 0.9},
    ),
    "Nesterov": (
        linear_classifier.nesterov_update,
        {"learning_rate": 0.03, "momentum": 0.9},
    ),
    "AdaGrad": (
        linear_classifier.adagrad_update,
        {"learning_rate": 0.4},
    ),
    "RMSProp": (
        linear_classifier.rmsprop_update,
        {"learning_rate": 0.02, "decay_rate": 0.99},
    ),
    "Adam": (
        linear_classifier.adam_update,
        {"learning_rate": 0.04},
    ),
}

results = {}
for name, (update_rule, config) in experiments.items():
    W, history = linear_classifier.train_linear_classifier(
        linear_classifier.svm_loss_vectorized,
        update_rule,
        data["X_train"],
        data["y_train"],
        optimizer_config=config,
        reg=1e-3,
        num_iters=250,
        batch_size=64,
        seed=5,
    )
    predictions = linear_classifier.predict_linear_classifier(
        W, data["X_val"]
    )
    validation_accuracy = linear_classifier.accuracy(
        predictions, data["y_val"]
    )
    early_loss = sum(history[:20]) / 20
    late_loss = sum(history[-20:]) / 20
    results[name] = (validation_accuracy, early_loss, late_loss)
    plt.plot(history, label=name, alpha=0.8)

plt.ylim(0, 0.8)
plt.xlabel("iteration")
plt.ylabel("minibatch loss")
plt.title("Optimizer comparison")
plt.legend()
plt.show()

for name, (accuracy, early_loss, late_loss) in results.items():
    print(
        f"{name:8s} accuracy={100 * accuracy:6.2f}% "
        f"early_loss={early_loss:.4f} late_loss={late_loss:.4f}"
    )
    assert late_loss < early_loss
    assert accuracy > 0.85


