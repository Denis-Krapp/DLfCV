1. For an operation out = a + b, the local derivative with respect to either input is 1 (d(out)/da = 1 and d(out)/db = 1). By the chain rule, multiplying the upstream gradient by 1 simply routes the upstream gradient unchanged to each incoming input.

2. For an operation out = a * b, the local derivative with respect to one input is the current value of the other input (d(out)/da = b and d(out)/db = a). Applying the chain rule scales the incoming upstream gradient by the opposite factor.

3. A small relative error (e.g., < 1e-7) confirms that the manual analytic backpropagation is mathematically correct and free of implementation bugs, as finite-difference numerical differentiation provides an independent approximation of the true derivative directly from the forward pass.