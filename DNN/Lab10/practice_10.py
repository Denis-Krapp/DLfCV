from pathlib import Path
import importlib
import sys

import torch

candidate_dirs = [
    Path.cwd(),
    Path.cwd() / 'Lab10',
    Path.cwd() / 'DNN' / 'Lab10',
    Path.cwd() / 'Computer Vision' / 'DNN' / 'Lab10',
]
lab_dir = next(
    (path.resolve() for path in candidate_dirs if (path / 'scalar_backprop.py').is_file()),
    None,
)
if lab_dir is None:
    raise FileNotFoundError(
        'Could not find scalar_backprop.py. Keep it together with the notebook.'
    )

if str(lab_dir) not in sys.path:
    sys.path.insert(0, str(lab_dir))
sys.modules.pop('scalar_backprop', None)

import scalar_backprop

importlib.reload(scalar_backprop)
torch.set_default_dtype(torch.float64)

print('PyTorch:', torch.__version__)
print('Lab directory:', lab_dir)
print('Implementation:', Path(scalar_backprop.__file__).resolve())


values = scalar_backprop.make_fixture()
for name, value in values.items():
    print(f'{name} = {value.item():+.1f}')


importlib.reload(scalar_backprop)

loss, cache = scalar_backprop.scalar_forward(values)
for name in ('s0', 's1', 's2', 's3', 'loss'):
    print(f'{name}: {cache[name].item():+.6f}')

assert torch.allclose(cache['s0'], torch.tensor(-2.0))
assert torch.allclose(cache['s1'], torch.tensor(6.0))
assert torch.allclose(cache['s2'], torch.tensor(4.0))
assert torch.allclose(cache['s3'], torch.tensor(1.0))
assert torch.allclose(loss, torch.tensor(0.7310585786300049))
print('Forward-pass check passed.')


importlib.reload(scalar_backprop)

loss, cache = scalar_backprop.scalar_forward(values)
grads = scalar_backprop.scalar_backward(cache)
for name, gradient in grads.items():
    print(f'dL/d{name}: {gradient.item():+.6f}')

expected_grads = {
    'w0': torch.tensor(-0.19661193324148185),
    'x0': torch.tensor(0.3932238664829637),
    'w1': torch.tensor(-0.3932238664829637),
    'x1': torch.tensor(-0.5898357997244456),
    'w2': torch.tensor(0.19661193324148185),
}
for name, expected in expected_grads.items():
    assert torch.allclose(grads[name], expected)
print('Manual-backpropagation check passed.')


importlib.reload(scalar_backprop)

report = scalar_backprop.gradient_report(values)
for name, (numerical, analytic, error) in report.items():
    print(
        f'{name}: numerical={numerical.item():+.8f}  ' 
        f'analytic={analytic.item():+.8f}  relative_error={error:.3e}'
    )

maximum_error = max(error for _, _, error in report.values())
print('maximum relative error:', maximum_error)
assert maximum_error < 1e-7


importlib.reload(scalar_backprop)

autograd_values = {
    name: value.detach().clone().requires_grad_(True)
    for name, value in values.items()
}
reference_loss, _ = scalar_backprop.scalar_forward(autograd_values)
reference_loss.backward()

_, cache = scalar_backprop.scalar_forward(values)
manual_grads = scalar_backprop.scalar_backward(cache)
for name in manual_grads:
    autograd_gradient = autograd_values[name].grad
    print(
        f'{name}: manual={manual_grads[name].item():+.8f}  ' 
        f'autograd={autograd_gradient.item():+.8f}'
    )
    assert torch.allclose(manual_grads[name], autograd_gradient)
print('Autograd comparison passed.')