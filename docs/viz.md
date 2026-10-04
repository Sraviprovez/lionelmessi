# Visualizations

Every plotting function accepts an optional Matplotlib `ax` and returns it, so
plots compose freely. `save_plot(ax, path, dpi=300)` writes a high-resolution
figure. If `mplsoccer` is installed it is used for pitch geometry; otherwise a
native pitch is drawn.

| Function | Description |
| --- | --- |
| `plot_pitch` | Base StatsBomb-oriented pitch |
| `plot_xt_surface` | Expected Threat surface with colourbar |
| `plot_shot_map` | Shots scaled by xG, coloured by outcome |
| `plot_goal_map` | Goals only |
| `plot_assist_map` | Assist origins with arrows |
| `plot_pass_map` | Passes as arrows (completed / progressive filters) |
| `plot_pass_network` | Coarse zone network of pass counts |
| `plot_carry_map` | Carries as arrows |
| `plot_heatmap` | Touch density heatmap |
| `plot_season_heatmap` | Season × pitch-third heatmap |
| `plot_continuation` | A single possession chain |
| `plot_goal_breakdown` | The full chain leading to a goal |
| `plot_career_timeline` | Cumulative goals and assists |
| `plot_rolling_form` | Goals + assists with a rolling mean |

## Example

```python
import lionelmessi as lm

events = lm.load_events()
ax = lm.viz.plot_shot_map(lm.metrics.shot_map_data(events))
lm.viz.save_plot(ax, "shots.png")
```
