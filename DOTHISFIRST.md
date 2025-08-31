# Refactor plugin classes

## Goal
Move remaining plugin classes out of `marble/marblemain.py` into dedicated modules under `marble/plugins` with direct implementations (no wrappers) and adjust imports.

## Steps
1. For each neuron plugin still defined in `marble/marblemain.py` (ConvTranspose1D/2D/3D, MaxPool1D/2D/3D, Unfold2D, Fold2D, MaxUnpool1D/2D/3D), copy the full class implementation into the corresponding module in `marble/plugins/` and remove the class definition from `marble/marblemain.py`. [complete]
2. Update these plugin modules to include all necessary imports (typing, math, reporter helpers, and base mixins) so they are self-contained. [complete]
3. Replace wrapper modules for `wanderer_hyperevolution.py` and ensure the `HyperEvolutionPlugin` implementation lives entirely inside that file. [complete]
4. Remove duplicate plugin class definitions from `marble/marblemain.py` for wanderer and brain-training plugins (`L2WeightPenaltyPlugin`, `ContrastiveInfoNCEPlugin`, `TDQLearningPlugin`, `DistillationPlugin`, `WarmupDecayTrainPlugin`, `CurriculumTrainPlugin`, `BestLossPathPlugin`, `AlternatePathsCreatorPlugin`, `HyperEvolutionPlugin`, `BaseNeuroplasticityPlugin`). Import their classes from the plugin modules instead and keep registration calls where necessary. [complete]
5. After each removal, adjust `__all__` and registration logic so that plugin classes remain accessible via `marble.marblemain`. [complete]
6. Create a new plugin module for `BaseNeuroplasticityPlugin` (e.g., `marble/plugins/neuroplasticity_base.py`) and move its implementation there with a `register_neuroplasticity_type` call. [complete]
7. Run the full test suite (`py -3 -m unittest -v tests.<module>`) to ensure all functionality remains intact, focusing on tests covering convolution, pooling, unpooling, and wanderer plugins. [complete]
8. Update `ARCHITECTURE.md` to document that plugin implementations now reside in their own modules rather than in `marble/marblemain.py`. [complete]

# Add new plugin suites

## Goal
Introduce suites of five cutting-edge, highly experimental plugins for each
existing plugin type. Every plugin must expose all of its parameters via the
`expose_learnable_params` decorator and explore ideas that have no equivalent
in current ML literature.

## Steps
1. Enumerate existing plugin types in the repository. [complete]
2. Implement advanced neuron plugin suite. [complete]
3. Implement advanced synapse plugin suite. [complete]
4. Implement advanced wanderer plugin suite. [complete]
5. Implement advanced brain_train plugin suite. [complete]
6. Implement advanced selfattention plugin suite. [complete]
7. Implement advanced neuroplasticity plugin suite. [complete]

## Pending tests [complete]

All listed test modules have been executed and their outputs analyzed for
logical consistency.

# Add ultra plugin suites

## Goal
Introduce another wave of five ultra‑experimental plugins for every existing
plugin type. Each plugin must surface all learnable parameters through
`expose_learnable_params` and pursue emergent behaviour far beyond current
literature.

## Steps
1. Implement ultra neuron plugin suite. [complete]
2. Implement ultra synapse plugin suite. [complete]
3. Implement ultra wanderer plugin suite. [complete]
4. Implement ultra brain_train plugin suite. [complete]
5. Implement ultra selfattention plugin suite. [complete]
6. Implement ultra neuroplasticity plugin suite. [complete]

## Pending tests [complete]
Run dedicated tests for each ultra plugin suite to confirm registration and
learnable parameter exposure.

# Lock-based thread safety and immutable training

## Goal
Guarantee that `Brain` instances and their associated `Graph` structures are
never copied or deep-copied during training and that any thread safety is
enforced strictly through `Lock` primitives.

## Steps
1. Audit `marble/training.py`, `marble/graph.py`, and all plugin modules for
   any use of `copy`, `deepcopy`, or cloning of brain and graph objects during
   training. Replace such patterns with direct references. [complete]
2. Introduce explicit `threading.Lock` guards around training sections that
   require thread safety, removing any copy-based safety mechanisms. [complete]
3. Extend tests to assert object identity for brain and graph instances before
   and after training steps to detect accidental copies. [complete]
4. Document the lock-only policy and immutability guarantees in
   `ARCHITECTURE.md` and update tutorials if necessary. [complete]

# Pending tests

The following test modules still need to be run and outputs analyzed:
- tests/test_new_wanderer_plugins.py [complete]
- tests/test_parallel.py [complete]
- tests/test_plugin_stacking.py [complete]
- tests/test_quantumtype_plugin.py [complete]
- tests/test_reporter.py [complete]
- tests/test_reporter_clear.py [complete]
- tests/test_reporter_subgroups.py [complete]
- tests/test_selfattention_conv1d.py [complete]
- tests/test_super_advanced_neuron_plugins.py [complete]
- tests/test_synapse_plugins.py [complete]
- tests/test_training_with_datapairs.py [complete]
- tests/test_triple_contrast_plugin.py [complete]
- tests/test_ultra_brain_train_plugins.py [complete]
- tests/test_ultra_neuron_plugins.py [complete]
- tests/test_ultra_neuroplasticity_plugins.py [complete]
- tests/test_ultra_selfattention_plugins.py [complete]
- tests/test_ultra_synapse_plugins.py [complete]
- tests/test_ultra_wanderer_plugins.py [complete]
- tests/test_unfold_fold_unpool_improvement.py [complete]
- tests/test_wanderer.py [complete]
- tests/test_wanderer_alternate_paths_creator.py [complete]
- tests/test_wanderer_bestpath_weights.py [complete]
- tests/test_wanderer_walk_summary.py [complete]
- tests/test_wanderer_wayfinder_plugin.py [complete]

# SelfAttention metrics integration

## Goal
Ensure all SelfAttention routines consume the full set of reported state metrics
(`sa_loss`, `sa_loss_speed`, `sa_loss_accel`, `sa_model_complexity`) when making
decisions.

## Steps
1. Audit existing selfattention plugins and list those not using `ctx` metrics.
2. Update each plugin to adjust behaviour based on the metrics and log their
   usage to `REPORTER`.
3. Expand tests to cover metric-driven behaviour for every plugin.
4. Document the available metrics and integration guidelines in `ARCHITECTURE.md`.
