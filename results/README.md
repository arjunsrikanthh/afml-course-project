# Results

This is one project with two retained benchmark records:

- **Baseline** — the earlier 23-method comparison, retained as a reference point.
- **Final** — the primary 33-method benchmark used for the project's final analysis.
- **Synthesis** — comparisons calculated from the saved baseline and final forecasts.

The old run labels were `reference` and `expanded`. The repository now uses `baseline` and `final` for its user-facing paths and commands. The original release archive keeps its historical internal names so existing replay tools and published checksums remain valid.

| What you need | Where to find it |
| --- | --- |
| Baseline frozen protocol | [`baseline/frozen_protocol.json`](baseline/frozen_protocol.json) |
| Final frozen protocol | [`final/frozen_protocol.json`](final/frozen_protocol.json) |
| Combined comparison tables | [`synthesis/`](synthesis/) |
| Full original baseline outputs | [23-method archived run](https://github.com/arjunsrikanthh/afml-course-project/tree/8f184132654470d36f3c7af2ddd8df8a1ef660d8/results/reference) |
| Full original final outputs | [33-method archived run](https://github.com/arjunsrikanthh/afml-course-project/tree/8f184132654470d36f3c7af2ddd8df8/results/expanded) |
| Forecasts, fitting logs, saved weights and original source workbooks | [Frozen study release](https://github.com/arjunsrikanthh/afml-course-project/releases/tag/study-v1.0.0) |
| Follow-up and extension findings | [Follow-up](../followup/results/) · [Extension](../followup/extension/results/) |

To rerun the benchmarks, follow the [reproduction instructions](../docs/REPRODUCE.md).
