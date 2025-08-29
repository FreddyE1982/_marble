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
Introduce five new advanced plugins for each plugin type, exposing all parameters via `expose_learnable_params`.

## Steps
1. Enumerate existing plugin types in the repository. [complete]
2. Implement neuron plugin suite (Swish, Mish, GELU, SoftPlus, LeakyExp) and register them with tests and docs. [complete]
3. Implement synapse plugin suite. [complete]
4. Implement wanderer plugin suite.
5. Implement brain_train plugin suite.
6. Implement selfattention plugin suite.
7. Implement learning_paradigm plugin suite.
8. Implement neuroplasticity plugin suite.

## Pending tests

The following test modules still need to be executed and their outputs
analyzed for logical consistency:

- tests/test_advanced_neuron_plugins.py
- tests/test_autolobeplugin.py
- tests/test_autoneuron.py
- tests/test_autoplugin_exclude.py
- tests/test_autoplugin_explicit.py
- tests/test_batch_training_plugin.py
- tests/test_brain.py
- tests/test_brain_snapshot.py
- tests/test_brain_sparse.py
- tests/test_brain_sparse_io.py
- tests/test_codec.py
- tests/test_conv2d_conv3d_improvement.py
- tests/test_conv_improvement.py
- tests/test_conv_transpose_improvement.py
- tests/test_curriculum_and_temp_plugins.py
- tests/test_datapair.py
- tests/test_earlystop_plugin.py
- tests/test_epochs.py
- tests/test_findbestneurontype_fallback.py
- tests/test_graph.py
- tests/test_latent_and_synthetic_plugins.py
- tests/test_learnable_params.py
- tests/test_learning_paradigm.py
- tests/test_learning_paradigm_helpers.py
- tests/test_learning_paradigm_stacking.py
- tests/test_learning_paradigm_toggle_and_growth.py
- tests/test_lobe_training.py
- tests/test_maxpool_improvement.py
- tests/test_neuroplasticity.py
- tests/test_new_paradigms_and_plugins.py
- tests/test_new_wanderer_plugins.py
- tests/test_parallel.py
- tests/test_plugin_stacking.py
- tests/test_quantumtype_plugin.py
- tests/test_reporter.py
- tests/test_reporter_clear.py
- tests/test_reporter_subgroups.py
- tests/test_selfattention_conv1d.py
- tests/test_synapse_plugins.py
- tests/test_training_with_datapairs.py
- tests/test_triple_contrast_plugin.py
- tests/test_unfold_fold_unpool_improvement.py
- tests/test_wanderer.py
- tests/test_wanderer_alternate_paths_creator.py
- tests/test_wanderer_bestpath_weights.py
- tests/test_wanderer_helper_and_synapse.py
- tests/test_wanderer_walk_summary.py
- tests/test_wanderer_wayfinder_plugin.py
