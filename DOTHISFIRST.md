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
5. Implement advanced brain_train plugin suite.
6. Implement advanced selfattention plugin suite.
7. Implement advanced neuroplasticity plugin suite.

## Pending tests [complete]

All listed test modules have been executed and their outputs analyzed for
logical consistency.

# Lock-based thread safety and immutable training

## Goal
Guarantee that `Brain` instances and their associated `Graph` structures are
never copied or deep-copied during training and that any thread safety is
enforced strictly through `Lock` primitives.

## Steps
1. Audit `marble/training.py`, `marble/graph.py`, and all plugin modules for
   any use of `copy`, `deepcopy`, or cloning of brain and graph objects during
   training. Replace such patterns with direct references.
2. Introduce explicit `threading.Lock` guards around training sections that
   require thread safety, removing any copy-based safety mechanisms.
3. Extend tests to assert object identity for brain and graph instances before
   and after training steps to detect accidental copies.
4. Document the lock-only policy and immutability guarantees in
   `ARCHITECTURE.md` and update tutorials if necessary.
