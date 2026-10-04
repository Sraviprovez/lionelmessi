# Quickstart

## Install

```bash
pip install lionelmessi            # core
pip install "lionelmessi[viz,ml]"  # + mplsoccer/seaborn and scikit-learn
```

## Fetch and cache

```python
import lionelmessi as lm

events = lm.load_events()      # first run downloads; later runs are offline
print(lm.career_summary(events))
```

From the command line:

```bash
lionelmessi fetch
lionelmessi summary
```

## First xT plot

```python
xt = lm.ExpectedThreat().fit(events)
ax = xt.plot(annotate=False)
ax.figure.savefig("xt.png", dpi=300)
```

Or:

```bash
lionelmessi xt-fit --out xt.npz
lionelmessi xt-plot --model xt.npz --out xt.png
```

## Possession chains

```python
chains = lm.build_continuations(events, xt_model=xt)
big = lm.filter_continuations(chains, actor="Messi", min_xt_gain=0.05, ended_in_goal=True)
print(lm.continuation_summary(big))
```
