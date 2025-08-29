0) Definitions & Precedence

Step / Substep / Group

Ein Step ist ein Eintrag aus developmentplan.md.

Ist er zu groß, wird er in Substeps zerlegt.

Groups bündeln Substeps zu einem Paket, das in einem Run komplett durchgezogen wird.

Kennzeichnung

Abgeschlossene Elemente: [complete] am Ende der Zeile.

Der nächste zu implementierende Step/Substep ist immer der erste, der nicht als [complete] markiert ist.

Kompatibilität

Alle Steps/Substeps/Groups müssen so umgesetzt werden, dass sie mit allen anderen bereits implementierten Teilen kompatibel sind.

Besonders relevant: developmentplan.md: 3. und developmentplan.md: 4.

Precedence (höchste zuerst)

Explizite User-Instruktion

Spezifische Ausnahme-Regeln (z. B. GUI-only, relevant tests only, QUICKMODE)

Allgemeine Regeln in dieser Datei

Rule Extension Policy

⚠️ Verbot: Der Agent darf niemals Stubs oder Stub-Module für bestehende Module erzeugen – nicht für Tests, nicht für irgendetwas.
Alle Teile, die Torch benötigen, müssen immer torch importieren und verwenden. Auch Tests!

1) Arbeitsreihenfolge

Wenn der Agent anfängt zu arbeiten, ist die Reihenfolge immer:

Prüfen, ob die Anfrage des Nutzers in einem Turn erledigt werden kann.

Falls nein → DOTHISFIRST.md erstellen und dort alle nötigen Schritte eintragen.

Schritte aus DOTHISFIRST.md abarbeiten (Vorgehensweise identisch zu developmentplan.md).

Hat der Nutzer keine spezifische Anfrage gegeben und es gibt auch in DOTHISFIRST.md nichts zu tun → developmentplan.md aufrufen.

Gibt es auch dort nichts → developmentplan.md erweitern (neue Steps hinzufügen).

2) Sequential Execution Contract

Schritte müssen in der Reihenfolge aus developmentplan.md abgearbeitet werden.

Keine Sprünge, kein Überspringen.

Ist ein Step zu groß → in Substeps zerlegen, ohne Vereinfachung.

Substeps werden zu ausführbaren Groups gebündelt:

Jede Group ist eine Einheit, die in einem Run vollständig erledigt wird.

Groups sind so groß wie machbar, nicht überlappend und decken alle Substeps ab.

3) Group Execution via Buttons

Wenn Steps in Substeps zerlegt werden, muss der Agent Groups bilden.

Groups werden dem Nutzer als klickbare Buttons präsentiert.

Jeder Button startet die Abarbeitung aller Substeps der Group in der vorgegebenen Reihenfolge.

Keine Substeps dürfen ausgelassen werden.

4) Environment & Dependencies

requirements.txt ist installiert – kein erneutes pip install -r requirements.txt nötig.

Änderungen am Code → requirements.txt aktualisieren, Konflikte vermeiden/lösen.

5) Config Policy (YAML = Single Source of Truth)

Alle konfigurierbaren Parameter gehören in config.yaml.

Neue Parameter nur aufnehmen, wenn sie:

Im Code voll genutzt werden,

und die algorithmische Logik beeinflussen.

Sicherstellen: Änderung im YAML → Änderung im Verhalten (end-to-end).

Pre/Post Checks (außer QUICKMODE)

Preflight (0.1)

Vor Hauptaufgabe: Scan auf ungenutzte YAML-Parameter.

Alle ungenutzten Parameter sofort mit Code verdrahten und wirksam machen.

Danach: genau einen Fehler aus FAILEDTESTS.md beheben.

Postflight (0.2)

Nach Hauptaufgabe: Scan erneut laufen lassen.

Alle ungenutzten Parameter einbauen.

6) Documentation Duties

yaml-manual.txt → vollständige Doku aller config.yaml-Parameter (Zweck, Range, Default, Effekte).

TUTORIAL.md → Schritt-für-Schritt-Anleitung mit realen Projekten + ausführbarem Code.

Beide Dateien sofort aktualisieren, wenn Änderungen passieren.

7) Testing Policy

Keine globalen Test-Runs.

Jeder Testfile wird einzeln ausgeführt.

Reihenfolge: test1 → fix → rerun bis grün → test2 …

Wenn Code geändert → immer pytest-Test erstellen/aktualisieren.

Failing Tests → FAILEDTESTS.md ergänzen.

Keine Tests manipulieren, um Fehler zu verstecken.

Determinismus: Seeds setzen, Side Effects isolieren.

8) Change Hygiene

Jede Änderung → CHANGELOG aktualisieren.

Architektur-Entscheidungen → ADR (Architecture Decision Record) mit Begründung, Alternativen, Konsequenzen.

9) Prohibitions

Keine Platzhalter, Stubs, Demo-Versionen.

Keine Vereinfachung oder Verkürzung bestehender Funktionen.

Keine Umgehung echter Fehler.

Keine Reihenfolgeänderung im Plan.

Keine Loops für Test-Orchestrierung.

10) Rule Extension Policy

Neue Regeln dürfen nur am Ende angehängt werden (append-only).

Regeln müssen:

Eindeutig helfen,

Keine bestehenden Regeln schwächen oder widersprechen,

Eine vollständige Rule Proposal Block enthalten:

Rule Proposal Block
Rule: <Regel in einem Satz>
Context: <wo gilt sie>
Problem: <welches Problem tritt ohne auf>
Benefit: <warum besser>
Non-Conflict Statement: <warum nichts oben widerspricht>
Verification: <wie wird’s überprüft>
Rollback Plan: <wie zurücknehmen>

11) Post-Run Checklist

Dependencies passen.

YAML-Scans (0.2) grün.

Codeänderungen dokumentiert (Tests, Manual, Tutorial).

Alle Tests per File grün oder dokumentiert.

requirements.txt aktuell.

Vollständige Files geliefert.

CHANGELOG/ADR gepflegt.

12) QUICKMODE

Nur wenn der User es explizit sagt. Erlaubt:

Dependency-Install überspringen,

Preflight 0.1 überspringen,

Nur relevante Tests laufen lassen.

Alles andere bleibt gleich.

13) Developmentplan.md & DOTHISFIRST.md

developmentplan.md:

Muss existieren, sonst erstellen.

Enthält alle mittelfristigen Steps, die ohne Zeitdruck abgearbeitet werden.

Immer erweitern, wenn nichts mehr zu tun ist.

DOTHISFIRST.md:

Wird erstellt, wenn eine spezifische Nutzeranfrage in einem Run nicht lösbar ist.

Enthält alle Schritte zur Erfüllung der Anfrage, gegliedert wie developmentplan.md.

Hat immer höchste Priorität bis abgearbeitet.

14) Repository Layout Rule

Rule: Alle Python-Source-Files liegen in src/.
Context: Einheitliches Importsystem.
Problem: Ohne klares Layout brechen Imports.
Benefit: Weniger Import-Fehler, klare Struktur.
Non-Conflict Statement: Bricht keine anderen Regeln.
Verification: Alle Module aus src/ importierbar.
Rollback Plan: Falls Probleme → Rule streichen, Files umziehen.


1. imports are allowed across modules. marblemain.py remains the primary entry/aggregation point. if you refactor anything ensure that the functionality / algorithm / math is not changed by the refactoring!
2. you are NEVER allowed to simplify, "cut down", "shorten" or "narrow down the scope" of ANY existing function, class, algorithm or calculation in any way
2.1 to read files you MUST use the tool "show", here are the available commands:


Additional Operating Rules (non-contradictory)

- After code or docs changes, document updates in `ARCHITECTURE.md` alongside the change.
- Prefer Windows-friendly commands and chunked file reads (`show`), avoiding large unpaginated outputs.
- Treat `pytest` as potentially unavailable; default to Python’s `unittest` runner when executing tests.
- Ensure the package installs cleanly (editable install) before running tests so `import marble` resolves.
- When CUDA is available (`torch.cuda.is_available()`), prefer CUDA tensors; otherwise use CPU.
- Add high-level convenience helpers additively; never remove or narrow existing APIs.
- Imports may be used where appropriate across the codebase; prefer clear module boundaries and avoid circular imports.

Testing and Execution Addenda (non-contradictory)

- Default test command on Windows: `py -3 -m unittest discover -s tests -p "test*.py" -v`.
- Prefer editable install via `py -3 -m pip install -e .` before tests. If the environment intermittently fails editable install but local source imports work, proceed to run tests and report the packaging issue without blocking.
 - Linux packaging policy: Always pass `--break-system-packages` when invoking pip installs in managed environments to bypass PEP 668 restrictions (e.g., `python3 -m pip install -e . --break-system-packages`). Prefer a venv when available, but do not block on system policy in CI.

SelfAttention and Wanderer parameters

- SelfAttention: A coordination class that hosts "self attention routines" (plugins) which can observe reportable state and request Wanderer setting changes after each step. Changes take effect at the next step.
- Reporter access for self attention routines is read-only. Routines receive a read-only view of `REPORTER` (query `item`, `group`, `dirtree`) and cannot write.
- Settings management: SelfAttention exposes `get_param(name)` and `set_param(name, value)` over the Wanderer's public attributes (non-underscore). `set_param` queues updates which the Wanderer applies at the beginning of the next step.
- Policy for new Wanderer parameters: whenever a new parameter/setting is added to `Wanderer`, it must be immediately made accessible through SelfAttention (get/set) so routines can control it starting with the next step.
- Per-walk summaries: After each walk, the Wanderer writes a summary (final loss, mean per-step loss, steps, visited, timestamp) to the reporter under `training/walks`.

SelfAttention analysis capability

- The SelfAttention class maintains a rolling, configurable history of step analyses so routines can reason over the effects of earlier parameter changes.
- Each history record includes: timestamp, per-step loss, current-walk loss, neuron/synapse counts, the set of parameter changes that were applied at the step start, and a snapshot of the Wanderer’s current public settings.
- Routines can access this via `selfattention.history(n)` and adjust strategy accordingly.
- High-level helpers: `run_training_with_datapairs` accepts a `selfattention` instance to wire step-wise control into the shared `Wanderer` used across all datapairs.

Convolutional neuron data policy (additive rule)

- Conv neurons (conv1d/conv2d/conv3d) must source their input data from the actual graph:
  - Parameter inputs: exactly 5 incoming synapses labeled with a `type_name` starting with `param` supply [kernel, stride, padding, dilation, bias] in deterministic order (by source position, else id). These are required at creation.
  - Data inputs: one or more additional incoming synapses labeled `data` provide the input signal. The neuron aggregates these to build the correct dimensional input (1D concatenation; 2D rows; 3D slices). If no `data` synapses exist, the neuron may fall back to the `input_value` provided by the `Wanderer` or its own tensor for backward compatibility.
  - Outputs: exactly 1 outgoing synapse is required.
  - Computation runs on CUDA when available; otherwise CPU.

  show [DATEI] [ZEILEN]            → zeigt erste ZEILEN der Datei, merkt sich Datei & Position
  show continue [ZEILEN]           → zeigt die nächsten ZEILEN der zuletzt angezeigten Datei
  show previous [ZEILEN]           → zeigt die vorherigen ZEILEN (geht im Verlauf zurück)
  show goto [ZIEL] [ZEILEN]        → springt zur Zeile (Nummer oder Regex) in der letzten Datei
  show length [DATEI]              → gibt die Zeilenanzahl der Datei aus
  show search [DATEI] [REGEX]      → sucht via Regex, gibt "ZEILENNUMMER: INHALT" aus
 show get [ZIEL] [VOR] [NACH]     → zeigt Kontext um ZIEL (in letzter Datei): VOR, ZIEL, NACH

Show Tool Setup and Usage (additive, non-contradictory)

- Location: use the per-user path `C:\Users\<User>\AppData\Local\Programs\show\show.py` (on this machine: `C:\Users\Standardbenutzer\AppData\Local\Programs\show\show.py`).
- UTF‑8 invocation (avoids cp1252 Unicode errors): `py -3 -X utf8 C:\Users\Standardbenutzer\AppData\Local\Programs\show\show.py <args>`.
- PowerShell helper (optional): add to your profile to call `show` directly:
  - `function show { param([Parameter(ValueFromRemainingArguments=$true)][string[]]$args) py -3 -X utf8 "$env:LOCALAPPDATA\Programs\show\show.py" @args }`
- Examples: `show ARCHITECTURE.md 200`, `show continue 200`, `show goto 85 40`, `show search AGENTS.md "^\\d+\\."`.
- Troubleshooting:
  - UnicodeEncodeError: include `-X utf8` in the Python invocation.
  - Not found: verify the path exists with `Get-Item "$env:LOCALAPPDATA\Programs\show\show.py"` or supply the correct location.
  - Policy: per rule 2.1, always use this tool for file reads; prefer chunked paging to avoid large unpaginated outputs (Windows-friendly).
  - Fallback policy: If the `show` tool is not available or cannot be executed, assume we are on Linux and use Linux commands only for file reads until `show` becomes available again.

Learnable Parameters and SelfAttention Policy (additive)

- Per‑Neuron Learnables: Any parameter that can be learnable in a neuron type (e.g., convolution kernels, plugin‑specific biases, and tunable hyperparameters like kernel_size/stride/padding/dilation) MUST be exposed as a per‑neuron learnable parameter.
- SelfAttention‑Only Optimization: All optimization capabilities for these learnable parameters are implemented via the `SelfAttention` class. Routines decide if/when/how to optimize per‑neuron params; Wanderer only wires the hooks for updates after backprop.
- Accessibility: SelfAttention exposes helpers to register and control learnables per neuron (ensure/create, enable/disable optimization, per‑param LR). Plugins read learnables from the neuron's ` _plugin_state['learnable_params']` if present.
- Defaults: If no SelfAttention routine registers a learnable, plugins fall back to values provided by existing parameter neurons to preserve backward compatibility.

Wanderer Plugin Policy (additive)

- Full algorithm override: Wanderer plugins MAY completely replace the base wandering algorithm by exposing a `walk(wanderer, max_steps, start, lr, loss_fn)` method. If present, `Wanderer.walk` delegates to it.
- Backward compatibility: Existing hooks `on_init`, `choose_next`, and `loss` remain supported. If no override is provided, the base algorithm runs.
- Reporting: When a plugin override is used and returns a dict with `{loss, steps, visited}`, the Wanderer records a per-walk summary (`training/walks`) for consistency.

Learning Paradigm Plugins (additive)

- Purpose: High-level plugins loaded into a `Brain` to compose and coordinate existing plugin systems (Wanderer, SelfAttention, neuron/synapse/brain-trainer plugins) into a new learning paradigm.
- Registry and loading: register via `register_learning_paradigm_type(name, plugin)`; load with `Brain.load_paradigm(name, config)`.
- Capabilities: May use any available plugin hooks and APIs to create/attach routines, set/get parameters, register new types, or override Wanderer behavior (via Wanderer plugin override).
- Wiring: Training helpers (`run_training_with_datapairs`) detect loaded paradigms and call `on_wanderer(wanderer)` so a paradigm can attach SelfAttention or otherwise configure the active Wanderer instance. Wanderer also invokes paradigm hooks directly, supporting neuroplasticity-like `on_init`, `on_step`, and `on_walk_end` so paradigms can perform any actions neuroplasticity plugins can.
- Toggling: Paradigms can be enabled/disabled at runtime without unloading (e.g., `brain.enable_paradigm(obj_or_name, enabled=False)`). Only enabled paradigms receive hooks and are applied by helpers.
- Stacking: Multiple learning paradigms can be loaded simultaneously; all hooks are executed additively in load order.
- Temporary modification: Use `push_temporary_plugins(wanderer, wanderer_types=[...], neuro_types=[...])` to temporarily inject plugin stacks for the duration of a run; restore with `pop_temporary_plugins(wanderer, handle)`.

Plugin Stacking Policy (additive)

- All plugin classes support stacking in an additive manner:
  - Wanderer: pass multiple plugin names as a comma-separated string or list. All `on_init` and `after`-style hooks run; `choose_next` composes with last valid choice winning; `loss` contributions sum when any are provided; if multiple `walk` overrides exist, the first wins (others ignored) for determinism.
  - Brain training: pass multiple trainer names similarly; hooks compose/merge (last wins on conflicts for overrides; all `after` hooks run).
  - SelfAttention: already stackable via routines list. Continue using multiple routines safely.
  - Neuron/Synapse types remain single-per-object by design; stacking applies to orchestrator-style plugins (Wanderer, Brain training, SelfAttention) that expose composable hooks.

3. you have to TEST after every change and fix errors by changing the actual code NOT by changing tests
4. tests are NOT allowed to monkeypatch in anyway or form
5. the project has to FULLY work as package, test that and correct problems!
6. if torch is needed you may ONLY install the CPU version!
7. you are on WINDOWS, so you can use WINDOWS shell commands only, you can use "sed" though since a windows version of it has been installed and put into path
8. if a GPU is present and torch reports CUDA available, GPU must be used for everything (prefer CUDA tensors; otherwise fall back to CPU)
9. you are NEVER allowed to remove ANY existing part of functionality, algorithm or calculation. all of users prompts are to be considered ADDITIVE unless the user
   specifies "\[REMOVE-FORCE]" in his prompt!
10. periodically ADD high level (convienience) functions!
11. reread this AGENTS.md after every prompt from the user
12. the agent is to add new rules to this AGENTS.md file to speed up and improve its work, but the new rules the agent adds MUST NOT contradict or modify any existing rule
    the agent may NOT remove existing rules
13. the agent creates / updates a file "ARCHITECTURE.md" in which it keeps updated descriptions of the projects architecture
14. always run the full test suite after any repository write (docs or code)
15. document any new behavior or module in ARCHITECTURE.md as part of the change
16. prefer dependency injection where it simplifies coupling; imports across modules are allowed.
17. when CUDA is available, place models/inputs/targets on GPU by default
18. prefer running tests via `py -3 -m pytest -q` on Windows to ensure the correct interpreter
19. when reading large files in tooling, page or chunk outputs to avoid truncation; use PowerShell `Get-Content -Raw` for reliability
20. after any repo write (docs/code/packaging), immediately run the full test suite and fix failures by changing code, not tests
21. ensure the project is installable as a package; validate with `py -3 -m pip install -e .` and re-run tests using the installed package
22. imports are allowed in any file; maintain clean boundaries and avoid unnecessary dependencies.
23. when CUDA is available, default-create tensors and move inputs to CUDA unless explicitly overridden
24. document any added high-level convenience APIs in ARCHITECTURE.md at the time of change
25. missing dependencies shall be installed by the agent automatically. If installation fails, the agent must inform the user which dependency is needed and ask the user to install it and report back once done.
26. testing reliability: on Windows, prefer `py -3 -m pytest -q -s` to run tests (the `-s` flag ensures that test `print` output is shown). If pytest invocation is flaky or unavailable, fall back to `py -3 -m unittest -v tests.<module_name>` for each test module.
27. before running tests, ensure the package is installed/editable with `py -3 -m pip install -e .` so `import marble` resolves consistently during tests.
28. all tests must print concise, relevant progress/output to the console (e.g., counts, key results, or stats) without excessive verbosity, to aid debugging in CI and local runs.
29. after running tests, analyze the printed output to ensure it makes logical sense; if inconsistencies or illogical results appear, fix the underlying code under test (not the tests) to resolve the issues.
30. all code must be reportable: use the global reporter to log relevant data, organized into logical groups and subgroups, so behavior can be audited from tests and during development.
31. all tests must use the reporter to log key checkpoints and metrics; tests should assert behavior and also record useful state via `REPORTER` for later analysis.
32. patch discipline: before editing large files (especially `marble/marblemain.py`), locate precise anchors (e.g., class/function headers, `__all__` blocks) with a search, then apply a single, targeted patch; avoid interleaving many small patches that touch distant regions.
33. when a patch fails (expected lines not found), first fetch and inspect surrounding context to adjust anchors, then reattempt with corrected locations—do not blindly retry.
34. group related changes into one coherent patch per file; batch code + docs updates together when practical to minimize iterations.
35. validate syntax immediately after edits using `py -3 -m py_compile <changed files>` before running tests, and prefer quick unit invocations for the directly affected modules.
36. respect Windows shell constraints: avoid heredocs (`<<`), prefer `apply_patch` for file changes and small Python runner files when inline execution is needed.
37. keep patches minimal, idempotent, and avoid duplication (e.g., do not reinsert existing branches like `elif` blocks); ensure control flow and indentation remain valid.
38. for long functions, prefer adding well-delimited helpers over fragmenting logic; maintain existing public APIs and behavior while extending per project rules.
39. always re-run the full suite (`py -3 -m pytest -q -s`) after successful syntax checks; analyze printed outputs for logical correctness and fix code (not tests) on inconsistencies.
40. document major edits in ARCHITECTURE.md contemporaneously to keep anchors stable and reduce future patch churn.

41. When adding data wrappers (e.g., DataPair), prefer placing them in cohesive modules; imports are allowed. Keep dependency injection where it improves testability.
42. Always log encode/decode events for new data utilities under an appropriate REPORTER group (e.g., `datapair/events`) to support test auditability.
43. Prefer convenience helpers alongside new classes (e.g., `make_datapair`, `encode_datapair`, `decode_datapair`) without removing or narrowing any existing APIs.
44. Test runner policy: Treat `pytest` as unavailable/unstable in this environment; the agent SHALL run tests using `unittest` modules by default (leveraging rule 26’s fallback). Only use `pytest` if the user explicitly requests it and stability is verified.
45. Neuroplasticity: Add new capabilities conservatively with default plugins that are safe and minimally invasive. Hook them into `Wanderer` via plugin registries and ensure all actions are logged via `REPORTER`.
46. Clear reporter state between tests using `Reporter.clear_group` (via `clear_report_group`) to avoid cross-test interference.
47. Helper functions that read reporter data must handle missing groups gracefully and return `None` rather than raising exceptions.
48. `run_training_with_datapairs` defaults to streaming mode, must not materialize full datasets, and must drop each consumed sample while ensuring the Brain stores snapshots during training.
49. Progress reporting must use `tqdm`, automatically choosing `tqdm.notebook` when running inside IPython environments.
Temporary Test Deferral (additive, non-contradictory)

- HyperEvolution comparison test: Deferred for now to keep CI time stable and avoid flakiness while architecture-search behavior evolves. This does NOT remove or narrow any functionality; it only defers a heavy integration test. Re-enable once stable budgets (pairs/steps/epochs) are agreed.
- All other tests remain required and must pass. HyperEvolution may still be exercised manually via examples or targeted runs.
- Never simulate or approximate required objectives or training steps; implement full computations with true parameter updates. No “stubs”, no reduced or proxy losses.
- Do not import third‑party model architectures or optimizers. Build encoders/decoders/transformers and optimizers in‑house using torch tensor ops only (no torch.optim, no nn.Module layers). Keep all code in `marble/marblemain.py`.

Warning and Backend Policy (additive)

- Never silence, filter, or ignore warnings/logs to “make them disappear.” Instead, fix the root cause so warnings are not triggered in the first place.
- Disable unsupported backends via configuration rather than relying on noisy runtime probing. Example: set `PYTORCH_DISABLE_NNPACK=1` before any torch import to avoid NNPACK initialization warnings on unsupported CPUs. This is applied in `marble/__init__.py`.
- Prefer pure-Python compute paths on CPU when high-performance torch backends would initialize unsupported CPU engines (e.g., NNPACK) and emit warnings. Use torch accelerated paths on CUDA only. This keeps behavior correct while preventing backend-init warnings on constrained hardware.
 - When creating Python scalars from tensors that may require grad (e.g., to initialize autograd parameters), always detach to CPU before conversion (`tensor.detach().to('cpu').item()`), avoiding PyTorch warnings while preserving gradient flow on the new autograd tensors.

Hugging Face Datasets Policy (additive)

- Default streaming: When loading Hugging Face datasets via project helpers, prefer `streaming=True` by default to minimize memory and start processing early.
- Auto-encoding fields: Any field accessed from a Hugging Face dataset example returned by project helpers is automatically encoded through `UniversalTensorCodec` before being returned, honoring CUDA preference when available.
- Token sourcing: `hf_login` reads the token from the explicit parameter or from `HF_TOKEN`/`HUGGINGFACE_TOKEN` environment variables. No interactive prompts are used in tests.
- Lazy imports: Hugging Face libraries are imported lazily inside helpers in `marble/marblemain.py` to keep import-time requirements minimal; missing dependencies are reported clearly when a helper is used.
- Example scripts fetching Hugging Face datasets must use `load_hf_streaming_dataset` and comment which dataset fields are consumed.

50. Utility scripts for repository setup (e.g., `clone_or_update.sh` and `.ps1`) must remain in the repo root, use `git -C` for pulls, and perform editable installs via `pip install -e .`.
51. Example scripts must reference only existing plugins and paradigms so they run without missing-component errors.
52. Before running any tests, explicitly install the CPU-only version of torch via `pip install --index-url https://download.pytorch.org/whl/cpu torch`.
53. Example scripts that enable batching must load the `batchtrainer` Wanderer plugin and set `batch_size` consistently in both `neuro_config` and helper arguments.
54. Dynamic neuron types like `AutoNeuron` must revert to the previous type on errors and expose selection parameters via `expose_learnable_params` to keep gradients intact.
55. Development plan directories must contain a `developmentplan.md` with Step/Substep/Subsubstep structure so new modules can be executed sequentially.
56. Document analyses of external dependencies in corresponding subdirectories as markdown files to preserve context for future steps.
57. Modules under `3d_printer_sim` must remain self-contained, using only Python's standard library and other files within that directory. This restriction applies only to repository code; external dependencies installed via package managers are permitted when needed.
58. The microcontroller in `3d_printer_sim` must track pin-to-component mappings so tests can verify attached devices by pin number.
59. Motion modules in `3d_printer_sim` must update deterministically via time-step `update(dt)` functions and enforce configured acceleration and jerk limits validated by unit tests.
60. For work inside `3d_printer_sim`, do not install or rely on `torch`; other dependencies may be installed as required.
61. Visualization for the printer simulator must leverage a dedicated 3D framework (e.g., Unity, Three.js, or similar) so that every simulated component—including moving axes, print bed, and deposited filament—has a live visual counterpart.
62. After every change within `3d_printer_sim`, evaluate whether new tasks for visualization or full live 3D rendering are required. If so, update `3d_printer_sim/developmentplan.md` with the necessary steps.
63. Treat the `3d_printer_sim` directory as a fully distinct project. Repository rules that only concern other parts of the repo do not apply inside this folder unless they logically make sense for both projects.
64. All steps in `3d_printer_sim/developmentplan.md` must be implemented fully and exactly as written—no mocking, stubs, skeletons, or minimal implementations are ever permitted.
