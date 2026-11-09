1. The XOR arrangement is not linearly separable. Opposite diagonal quadrants share the same class label, which cannot be partitioned correctly by any single straight decision boundary (hyperplane) in a two-dimensional feature space.

2. Each hidden ReLU unit introduces a linear partition boundary. As hidden width increases, the network combines more piecewise-linear boundaries, allowing the decision region to carve out non-linear enclosed shapes, fit the diagonal quadrants accurately, and separate the XOR clusters.

3. Increasing weight decay penalizes large parameter values by adding an explicit L2 penalty to the objective. This shrinks the overall parameter L2 norm, keeping weights closer to zero and resulting in smoother, less abrupt transition regions.

4. PyTorch accumulates gradients into the `.grad` attributes by default instead of overwriting them. Calling `optimizer.zero_grad()` resets existing gradients to zero so that the new gradients computed by `loss.backward()` reflect only the current batch rather than accumulating on top of previous iterations.