# Models

## Expected Threat (xT)

The pitch is divided into an `x_bins × y_bins` grid (16 × 12 by default). For
each cell $i$ we estimate:

- $p_{shot}^{(i)}$ — the probability an action in the cell is a shot,
- $p_{goal}^{(i)}$ — the probability a shot from the cell scores, and
- a transition matrix $T$, where $T_{ij}$ is the probability a non-shot action
  started in cell $i$ ends in cell $j$.

Unlike a uniform prior, $T$ is estimated from **observed** pass and carry end
locations:

$$
T_{ij} = \frac{\text{moves}(i \to j)}{\text{moves}(i)}.
$$

Only cells with no observed moves fall back to a uniform distribution over all
$n$ cells.

### Value iteration

Cell values follow

$$
V_i = p_{shot}^{(i)} p_{goal}^{(i)} + \left(1 - p_{shot}^{(i)}\right) \sum_j T_{ij} V_j,
$$

solved by fixed-point iteration

$$
V^{(k+1)} = p_{shot} \odot p_{goal} + \left(1 - p_{shot}\right) \odot (T V^{(k)}),
$$

from $V^{(0)} = 0$, stopping when $\lVert V^{(k+1)} - V^{(k)} \rVert_\infty < 10^{-6}$
or after 100 iterations.

The iteration is a contraction with Lipschitz constant at most
$\max_i (1 - p_{shot}^{(i)}) \le 1$, so it converges (fast, because `p_shot`
is close to 1 near goal and `T` is substochastic on non-shot mass).

### API

```python
xt = lionelmessi.ExpectedThreat(x_bins=16, y_bins=12).fit(events)
xt.xt_surface          # ndarray, shape (y_bins, x_bins)
xt.xt_at(110, 40)      # value at a point
xt.xt((60, 20), (110, 40))  # threat gained
xt.save("xt.npz")
lionelmessi.ExpectedThreat.load("xt.npz")
```

## VAEP-style model

`ActionValueModel` learns $P(\text{goal} \mid \text{state})$ from frame-level
features (location, distance and angle to goal, pressure, shot flag) with a
logistic-regression classifier, and scores an action by the change in that
probability. It requires the `ml` extra.

## Pitch control

`pitch_control_surface` returns a simple ball-relative surface where control
decays with the approximate time for the ball to reach each cell.
