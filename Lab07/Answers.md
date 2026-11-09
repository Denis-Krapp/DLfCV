1. In steep directions where gradients oscillate back and forth and change sign between iterations, consecutive gradient vectors cancel each other out in the moving average velocity. At the same time, gradients in consistent directions add up and accelerate, smoothing out the trajectory and damping vertical bounces.

2. AdaGrad monotonically accumulates squared gradients into its running sum over all past iterations. Because every non-zero gradient term strictly increases this sum, the denominator grows indefinitely without decay, causing the effective step size to continuously shrink until updates prematurely stall.

3. - The first moment acts like momentum, computing an exponentially decaying moving average of past gradients to maintain direction and smooth out gradient noise.
   - The second moment tracks the moving average of uncentered squared gradients to scale coordinates individually, dampening steps along high-gradient dimensions and boosting progress along low-gradient ones.

4. Both moments are initialized to zero tensors. In early iterations, the decay factors keep the uncorrected estimates heavily biased toward zero. Without dividing by (1 - beta^t), the second moment remains artificially tiny, which would cause an unintended massive learning rate spike during initial steps.