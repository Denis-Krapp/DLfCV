from pathlib import Path
import importlib
import sys

import matplotlib.pyplot as plt
import torch

candidate_dirs = [
    Path.cwd(),
    Path.cwd() / 'Lab08',
    Path.cwd() / 'DNN' / 'Lab08',
    Path.cwd() / 'Computer Vision' / 'DNN' / 'Lab08',
]
lab_dir = next(
    (
        path.resolve()
        for path in candidate_dirs
        if (path / 'two_layer_net.py').is_file()
        and (path / 'dlcv2026').is_dir()
    ),
    None,
)
if lab_dir is None:
    raise FileNotFoundError(
        'Could not find the Lab08 support files. Keep two_layer_net.py '
        'and the dlcv2026 folder together with the notebook.'
    )

if str(lab_dir) not in sys.path:
    sys.path.insert(0, str(lab_dir))
for module_name in tuple(sys.modules):
    if (
        module_name == 'two_layer_net'
        or module_name == 'dlcv2026'
        or module_name.startswith('dlcv2026.')
    ):
        del sys.modules[module_name]

import dlcv2026
import two_layer_net

importlib.reload(two_layer_net)
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

print('PyTorch:', torch.__version__)
print('Lab directory:', lab_dir)
print('Implementation:', Path(two_layer_net.__file__).resolve())


fixture = dlcv2026.make_forward_fixture()
X = fixture['X']
params = fixture['params']

print('X shape:', tuple(X.shape))
for name, value in params.items():
    print(f'{name} shape:', tuple(value.shape))


importlib.reload(two_layer_net)

values = torch.tensor([[-2.0, 0.0, 1.5], [3.0, -0.5, -4.0]])
activated = two_layer_net.relu(values)
expected = torch.tensor([[0.0, 0.0, 1.5], [3.0, 0.0, 0.0]])

print('ReLU output:\n', activated)
assert torch.equal(activated, expected)
print('ReLU check passed.')


importlib.reload(two_layer_net)

scores, hidden = two_layer_net.two_layer_forward(params, X)
print('hidden activations:\n', hidden)
print('scores:\n', scores)

assert torch.allclose(hidden, fixture['expected_hidden'])
assert torch.allclose(scores, fixture['expected_scores'])
print('Forward-pass check passed.')


importlib.reload(two_layer_net)

predictions = two_layer_net.predict_two_layer(params, X)
print('predicted classes:', predictions.tolist())
assert torch.equal(predictions, fixture['expected_predictions'])
print('Prediction check passed.')


X_linear, W1_linear, W2_linear = dlcv2026.make_linear_equivalence_fixture()
two_linear_layers = (X_linear @ W1_linear) @ W2_linear
one_linear_layer = X_linear @ (W1_linear @ W2_linear)

print('maximum difference:', (two_linear_layers - one_linear_layer).abs().max().item())
assert torch.allclose(two_linear_layers, one_linear_layer)


importlib.reload(two_layer_net)

demo_params = dlcv2026.make_diamond_network()
grid_points, grid_x, grid_y = dlcv2026.make_input_grid()
grid_predictions = two_layer_net.predict_two_layer(demo_params, grid_points)

plt.figure(figsize=(6, 5))
plt.contourf(
    grid_x.numpy(),
    grid_y.numpy(),
    grid_predictions.reshape(grid_x.shape).numpy(),
    levels=[-0.5, 0.5, 1.5],
    colors=['#f4a6a6', '#9fcf9f'],
    alpha=0.85,
)
plt.axhline(0.0, color='black', linewidth=0.8)
plt.axvline(0.0, color='black', linewidth=0.8)
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Predictions from a fixed two-layer ReLU network')
plt.show()


probe_points = torch.tensor([[0.0, 0.0], [0.5, 0.25], [1.25, 0.25], [-0.5, -0.75]])
probe_scores, probe_hidden = two_layer_net.two_layer_forward(
    demo_params, probe_points
)

print('probe points:\n', probe_points)
print('hidden activations:\n', probe_hidden)
print('scores:\n', probe_scores)
print('predictions:', probe_scores.argmax(dim=1).tolist())