1. For every incorrect class that currently violates the margin, the total loss decreases by 0.5 / N (clamped at 0 if the violation disappears entirely). If no margins are violated for that example, the loss remains unchanged.

2. The margin depends on the difference between two scores: the incorrect-class score and the correct-class score. As a result, two separate weight columns are involved in the derivative:
   - The incorrect-class column receives a +X[i] gradient update to push its score down.
   - The correct-class column receives a -X[i] gradient update to push its score up.

3. Increasing delta raises the bar for safety: the model demands a wider gap between the correct score and every incorrect score. This makes the margin violation condition easier to trigger, causing more (example, incorrect class) pairs to contribute positive loss and non-zero gradients.