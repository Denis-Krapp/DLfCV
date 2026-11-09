1. The bias vector b1 represents parameters of the layer itself, which are shared across all training and test examples regardless of batch size N. Broadcom broadcasting automatically expands (H,) across all N batch elements during the addition (X @ W1 + b1), avoiding unnecessary memory usage.

2. It selects the index of the class corresponding to the highest scalar score in that row, representing the model's most confident predicted class label for that sample.

3. Linear transformations are closed under composition. Without a non-linear activation between them, chaining two linear mappings computes (X @ W1 + b1) @ W2 + b2 = X @ (W1 @ W2) + (b1 @ W2 + b2), which is mathematically equivalent to a single linear layer with equivalent weight W_eff = W1 @ W2 and bias b_eff = b1 @ W2 + b2.

4. Units 0 and 3. Unit 0 activates for positive x1 (value 0.5), and Unit 3 activates for negative x2 (value 0.75). Units 1 and 2 remain zero.