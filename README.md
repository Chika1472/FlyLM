# FlyLM

**Language modeling with a Drosophila connectome as the recurrent computational substrate.**

FlyLM investigates whether an experimentally reconstructed nervous system can provide a useful structural prior for sequence learning. A learned interface converts text tokens into currents at anatomically annotated sensory neurons. Activity propagates through the recorded connectome, and a second interface reads motor-neuron activity to predict the next token. The research question is how much language-relevant computation and temporal context can emerge from this constrained recurrent circuit, and how its behavior changes when neuronal dynamics become more biologically informed.

The project combines three ingredients: fixed anatomical connectivity, trainable synaptic strengths and neuronal dynamics, and a supervised language objective. Neuron identities and directed connections remain traceable to the source reconstruction. This makes it possible to examine activity, perturb identifiable pathways, and test whether a particular biological constraint contributes to the artificial task. Connectome-constrained task optimization has precedent in fly visual-system modeling; applying it to text introduces a different task and an artificial sensory interface. [Lappalainen et al. (2024)][lappalainen2024]

The anatomical substrate is **MaleCNS v1.0**, covering the adult male *Drosophila melanogaster* central nervous system. Its source publication is [Berg et al. (2026)][berg2026]. FlyLM uses the neurons selected by the notebook's `status == 'Traced'` filter and all released connections between them. “Full graph” throughout this README means this retained graph from the selected release, rather than every segment or every biological connection in the animal.

**Research status:** FlyLM is an experimental implementation. Language competence, biological memory fidelity, and benefits over conventional language models remain empirical questions. Text-to-sensory and motor-to-token mappings are learned engineering interfaces; they do not imply that flies possess a language system. The updated notebooks contain saved execution outputs, including partial training diagnostics and an inference example. These are not a completed comparative evaluation, so this README does not present them as benchmark results.

## FlyLM-2

FlyLM-2 extends the earlier byte-level, graded-dynamics model with **SentencePiece Unigram tokenization and mixed graded/spiking neuronal dynamics**. It preserves the graph, anatomical input/output selection, dialogue-level data split, and response-only training objective.

Sensory and motor neurons are themselves members of that graph, and recurrent connectivity is retained. The current token's sensory drive is held constant for four simulation ticks. During generation, the prompt is processed once; subsequent predictions use the generated tokens and recurrent neural state. There is no attention mechanism, external language model, prompt reinjection, or direct token-to-logit bypass.

| Property | FlyLM-1 | FlyLM-2 |
|---|---|---|
| Text representation | UTF-8 byte tokens | Train-fitted SentencePiece Unigram with byte fallback |
| Recurrent state per neuron | Membrane `v`, synaptic signal `s`, adaptation `a` | `v`, `s`, `a`, plus refractory counter `q` |
| Signal dynamics | Graded | Exact-type evidence selects spiking or graded dynamics; unmatched types default to spiking |
| Structural substrate | Retained MaleCNS graph | Same graph-selection policy |
| Training | Gradient-based sequence learning | TBPTT with surrogate gradients through spike events |
| Control condition | Earlier graded formulation | `neuron_mode='graded_control'` disables spiking throughout the graph |
| Checkpoint compatibility | Byte-model architecture | `malecns_mixed_sentencepiece_v2`; incompatible with byte-model checkpoints |

## Model specification

### Anatomical graph and input/output ports

The notebook downloads three official [MaleCNS v1.0 files][malecns-data]:

| File | Role | Expected bytes in the notebook |
|---|---|---:|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | Neuron identity, status, type, class, and superclass | 14,483,314 |
| `body-neurotransmitters-male-cns-v1.0.feather` | Neurotransmitter ground truth and predictions | 43,282,834 |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | Directed neuron-pair synapse counts | 1,051,241,946 |

After selecting `Traced` neurons, the notebook sorts them by `bodyId` and constructs a compressed sparse row (CSR) matrix with **postsynaptic neurons as rows and presynaptic neurons as columns**. Two passes over Arrow record batches first count incoming edges and then fill disk-backed arrays. Duplicate neuron pairs are merged by summing counts. Existing self-connections and disconnected components are retained; reachability analysis is diagnostic and does not prune the graph.

The checked-in annotation audit records **165,122 selected neurons**, with SHA-256 `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2`. This is a result for a particular annotation file and filter. It should not be substituted for the source publication's broader reconstruction count. The actual retained edge count is determined at extraction time and recorded in the graph manifest and run statistics.

Input neurons have superclass `cb_sensory`, `ol_sensory`, `vnc_sensory`, `sensory_ascending`, or `sensory_descending`. Output neurons have superclass `cb_motor` or `vnc_motor`. The input and output sets must be nonempty and disjoint, and at least one motor neuron must be reachable from the sensory set.

### Synaptic weights and neurotransmitter constraints

For an anatomical edge from neuron $j$ to neuron $i$, let $c_{ij}$ be its aggregated synapse count. The fixed count prior is

$$
p_{ij}=\frac{\log(1+c_{ij})}
{\sum_{k\rightarrow i}\log(1+c_{ik})}.
$$

The effective trainable weight is

$$
w_{ij}=p_{ij}
\begin{cases}
\eta_j\left(0.25+3.75\,\sigma(\theta_{ij})\right), & \eta_j\in\{-1,+1\},\\
4\tanh(\theta_{ij}), & \eta_j=0.
\end{cases}
$$

Recognized neurotransmitter ground truth takes precedence over a prediction; otherwise, the prediction must have confidence at least `0.8`. Acetylcholine supplies $\eta_j=+1$, and GABA supplies $\eta_j=-1$. Other transmitters and unknown assignments use $\eta_j=0$, allowing either sign. These are presynaptic sign approximations, not receptor-resolved conductance models. In particular, glutamate and dopamine are not assigned a universal excitatory, inhibitory, or reward meaning.

Every retained edge has a trainable scalar $\theta_{ij}$, while the count prior and topology stay fixed. Incoming normalization combines all incoming edges, including excitation and inhibition. Counts constrain the parameterization but do not establish physical synaptic efficacy. The logarithmic prior, bounds, and sign parameterization above are project-specific engineering choices.

### Four-state mixed dynamics

Each neuron maintains $(v,s,a,q)$: membrane state, nonnegative filtered synaptic output, slow adaptation, and remaining refractory ticks. The model state has shape `[4, batch, neurons]` and starts at zero. Neuronal parameters are shared by annotation `type`, falling back to `class`, then `superclass`, then an unknown group. **Parameter sharing and activity-mode assignment are separate:** the latter uses only exact `type` matches in the evidence registry.

For a simulation tick $t$, the input current and membrane candidate are

$$
I_i^t=\sum_{j\rightarrow i} w_{ij}s_j^t+u_i(x),\qquad
\widetilde v_i^{t+1}=v_i^t+\alpha_{m,i}
\left(-v_i^t+I_i^t-\beta_i a_i^t\right).
$$

Here $u(x)$ is nonzero only at the selected sensory ports, and $\beta_i$ is a learned sigmoid-bounded adaptation strength. Filter coefficients satisfy $\alpha=1-\exp(-1/\tau)$, with type-shared, bounded trainable $\tau$. The actual coupled recurrence is the explicit update above; the exponential coefficient does not make the whole coupled system an exact continuous-time simulation.

For graded neurons, release is

$$
r_i^{t+1}=\max\left(0,\tanh(\widetilde v_i^{t+1}-h_i)\right).
$$

For spiking neurons, an unblocked threshold crossing produces $z_i^{t+1}=1$; otherwise it produces zero. A spike resets the membrane to `0.0` and sets `q=2`. While refractory, the membrane is held at reset and spiking is suppressed. Both modes then update

$$
s_i^{t+1}=s_i^t+\alpha_{s,i}(r_i^{t+1}-s_i^t),\qquad
a_i^{t+1}=a_i^t+\alpha_{a,i}(r_i^{t+1}-a_i^t),
$$

using $r=z$ for spiking neurons. Thus synaptic output and adaptation continue to decay during refractoriness. This is a single-compartment, adaptive integrate-and-fire-style approximation alongside a graded branch, not a detailed ionic-channel model. [Burkitt (2006)][burkitt2006]

The forward spike is binary. Backpropagation substitutes the fast-sigmoid surrogate

$$
\frac{\partial z}{\partial g}\approx\frac{1}{(1+5|g|)^2},\qquad
g=\widetilde v-h.
$$

The reset decision and refractory counter are detached from differentiation. This relates to the surrogate-gradient literature and the voltage surrogate used in SuperSpike, but FlyLM-2 trains with TBPTT and Adam rather than implementing SuperSpike's full online learning rule. [Neftci et al. (2019)][neftci2019]; [Zenke and Ganguli (2018)][zenke2018]

| Parameter | Initialization | Allowed range or behavior |
|---|---:|---|
| Membrane time constant | 2 ticks | 0.5–32 ticks |
| Synaptic time constant | 2 ticks | 0.5–16 ticks |
| Adaptation time constant | 64 ticks | 8–512 ticks |
| Adaptation strength | 0.1 | Sigmoid parameterization, between 0 and 1 |
| Spiking threshold | 0.25 | `0.25 + 0.20 * tanh(type_parameter)` |
| Graded threshold | 0 | `0.5 * tanh(type_parameter)` |
| Spike reset / refractory duration | 0 / 2 ticks | Fixed configuration values |
| Simulation steps per token | 4 | Same token drive across all four ticks |

These numerical values are engineering assumptions. A tick is not a millisecond, and spike events per tick are not firing rates in hertz.

### Evidence-based activity assignment

The current registry, `flylm2_activity_evidence_2026-09-08_v2`, contains four publications and six exact annotation labels:

| MaleCNS `type` | Assigned mode | Evidence and transfer limitation |
|---|---|---|
| `H2` | Spiking | H2 electrophysiology in [Zhao et al. (2025)][zhao2025]; transfer across specimens |
| `DM1_lPN` | Spiking | DM1 projection-neuron recordings in [Gouwens and Wilson (2009)][gouwens2009]; female-to-male and anatomical-label correspondence |
| `HSN`, `HSE`, `HSS` | Graded | HS voltage recordings in [Pokusaeva et al. (2024)][pokusaeva2024]; electrical coupling studied there is absent from this model |
| `R1-R6` | Graded | Photoreceptor voltage recordings in [Juusola et al. (2016)][juusola2016]; pooled cell-family correspondence |
| Unmatched types | Spiking default | Engineering assumption; no physiological classification established by this registry |

This registry supports a model-mode choice, not transplantation of measured thresholds or time constants. No substring matching or broad neurotransmitter/class-based propagation of physiological labels is performed. Coverage and unmatched evidence types are exported for each run. The assigned spiking population includes both evidence-supported types and unmatched types assigned by default. Its size is not a measurement of the fly's biological spiking population; evidence coverage is reported separately.

The activity-assignment cell overrides the preliminary configuration with `CFG['unknown_activity'] = 'spiking'`. The mapper supports both defaults and preserves explicit graded evidence. It keeps `evidence_code=0` for unmatched types while writing their assigned execution code as `2` (spiking) in `dynamics_ids`; graded execution uses code `1`. The constructor therefore activates these unmatched neurons in mixed mode through `dynamics_ids == 2`. `neuron_mode='graded_control'` still forces every neuron to use graded dynamics. Inference restores saved execution codes rather than reassigning types from the current default.

Some introductory notebook text still describes the earlier graded default. The embedded standalone export source also retains a graded-only configuration validator; the supplied inference notebook replaces that validator before loading, allowing spiking-default checkpoints. A newly exported standalone module needs the same validator compatibility change before direct use. These compatibility details do not alter the new training assignment.

### Sensory encoding and motor decoding

Let $E\in\mathbb{R}^{V\times d}$ be the token embedding, with default $d=32$. A bias-free linear map projects $E[x]$ to sensory currents. After the last tick of each token, the model reads the motor subset of `s`, applies Layer Normalization with `eps=1e-6` and no learned affine parameters, and projects it to $d$ dimensions. Vocabulary logits use the same embedding matrix:

$$
\ell(x)=E\,W_{\mathrm{out}}\,
\mathrm{LayerNorm}(s_{\mathrm{motor}}).
$$

This implements input/output embedding tying. Relevant methodological references are [Ba et al. (2016)][ba2016], [Press and Wolf (2017)][press2017], and [Inan et al. (2017)][inan2017]. The readout is a compact engineered decoder; it does not reproduce biological motor behavior.

With $M$ retained edges, $G$ dynamics groups, $N_{\mathrm{in}}$ sensory ports, and $N_{\mathrm{out}}$ motor ports, the implementation has

$$
P=M+5G+dV+dN_{\mathrm{in}}+dN_{\mathrm{out}}
$$

trainable scalar parameters. The $5G$ term comprises three time constants, one threshold parameter, and one adaptation parameter per group. A small embedding dimension therefore does not imply a small model: individual edge parameters can dominate.

## Data, tokenization, and optimization

### Dialogue preparation

Training uses a **private Korean-language dataset**, supplied separately. Set `DATA_CFG['json_dir']` to its mounted directory. The parser expects JSON objects containing `document` arrays, with `utterance` entries providing `speaker_id` and `form`. It normalizes repeated whitespace, numbers speakers by first appearance, merges consecutive turns by the same speaker, and extracts adjacent speaker-1-to-speaker-2 pairs from exactly two-speaker dialogues. It removes duplicate documents and duplicate question–answer pairs, rejects conflicting duplicate document IDs, and excludes missing-speaker, single-speaker, and more-than-two-speaker cases.

Documents are split before tokenizer training using a seeded hash ordering, with a default validation fraction of `0.05`. All retained pairs from one document remain in the same split. Each training example is an adjacent question–answer pair, not the full accumulated conversation history.

### SentencePiece contract

SentencePiece learns a Unigram vocabulary from **training questions and answers only**. At most 100,000 training pairs are selected by a seed/hash rule; validation text and artificial speaker markers are excluded from vocabulary fitting. Default settings request 4,096 pieces, character coverage `0.9995`, all 256 byte-fallback pieces, identity normalization, no dummy prefix, and preservation of whitespace at the tokenizer stage. The earlier corpus whitespace normalization still applies. [Kudo and Richardson (2018)][sentencepiece2018]; [Kudo (2018)][kudo2018]

Special IDs are `UNK=0`, `BOS=1`, `EOS=2`, and `PAD=3`. A smaller corpus may produce fewer pieces than requested. The actual vocabulary size determines the embedding and output dimensions. The literal Unicode character `▁` is encoded through its UTF-8 bytes to distinguish it from SentencePiece's whitespace marker. Encoded examples use a `uint16` disk cache, and tokenizer metadata, preprocessing choices, and model hashes participate in cache identity.

```text
BOS + SP("화자1: " + question + "\n화자2: ") + SP(answer) + EOS
      prompt: no direct loss                 answer and EOS: supervised
```

Prompt and response are encoded separately to prevent a subword piece from crossing their boundary. The same prompt renderer is used for inference. Encoding is deterministic; the code uses the Unigram model family without enabling stochastic subword regularization.

### Objective and temporal credit assignment

Training uses teacher forcing and next-token cross-entropy. For the supervised answer/EOS positions $\mathcal{R}$ in a batch chunk,

$$
\mathcal{L}_{\mathrm{chunk}}=-\frac{1}{|\mathcal{R}|}
\sum_{(b,t)\in\mathcal{R}}\log p_\theta(y_{b,t}\mid x_{b,\le t}).
$$

Prompt targets and padding targets are masked with `-100`. Padding also freezes all four neural states for the affected batch row. Length-aware batching reduces padding, and rows retain independent state.

The early prompt is processed without gradients. Up to the final 32 prompt input tokens are included in the gradient-bearing phase with the response. Truncated backpropagation through time uses chunks of at most 64 tokens; an optimizer step occurs for each supervised chunk, and the returned state is detached before the next chunk. State values persist across chunks of the same example, so **64 tokens is a gradient horizon, not a hard inference-context limit**. State resets when a new batch of examples starts. [Williams and Peng (1990)][williams1990]

Optimization uses Adam with learning rate `0.001`, `foreach=False`, and global gradient-norm clipping at `1.0`. All model parameters, including existing edge strengths, dynamics, and adapters, are trainable. This is supervised global optimization, not local biological synaptic plasticity. [Kingma and Ba (2015)][adam2015]; [Pascanu et al. (2013)][pascanu2013]

### Sparse computation and memory

`SparseEdges` performs CSR sparse–dense multiplication in the forward pass. Its custom backward computes edge gradients from presynaptic activity and postsynaptic output gradients, then accumulates state gradients with indexed addition. Edges are processed in blocks of 262,144 to bound temporary allocations. Per-token activation checkpointing recomputes the four-tick transition during backward, trading computation for memory. [Chen et al. (2016)][chen2016]

The implementation avoids an $N\times N$ dense adjacency matrix, but still traverses the retained edges every tick. Recurrent message-passing cost scales approximately as $O(KBM)$ per token for $K$ ticks, batch size $B$, and $M$ edges. Sparse firing does not automatically reduce CSR work. Peak training memory also includes graph buffers, edge weights, gradients, Adam moments, and retained activations; it cannot be inferred from the four-state tensor alone.

## Running and evaluating FlyLM-2

### Obtaining the repository and model

Install Git LFS before cloning to retrieve the saved model and large graph arrays:

```bash
git lfs install
git clone https://github.com/Chika1472/FlyLM.git
cd FlyLM
git lfs pull
```

The private Korean-language dataset is not distributed in this repository. Local corpus examples, scratch runs, and generated caches are excluded from version control.

### Training workflow

1. Open the training notebook in a Kaggle GPU session, enable internet access, and attach the JSON corpus.
2. Verify `DATA_CFG['json_dir']`, then configure `CFG` and `TOK_CFG` before running subsequent cells.
3. Run cells from top to bottom. The notebook checks CUDA CSR support, prepares data and tokenizer caches, downloads the connectome, builds the graph, and assigns activity modes.
4. Run the included numerical checks and full-graph memory probe before training. The probe constructs a disposable model and performs a 64-token forward, backward, gradient clipping, and Adam step on synthetic tokens.
5. Preserve the run directory and exported bundle before the Kaggle session ends.

| Setting | Default |
|---|---|
| Target runtime | Kaggle P100, FP32 CUDA CSR |
| Seed / training batch size | `42` / `4` after the later training override (`2` in the initial configuration) |
| TBPTT / prompt gradient tail | `64` / `32` tokens |
| Maximum training budget | One epoch, capped at 2,000 optimizer updates by the later override (initial cap: 1,000) |
| VRAM budget fraction | `0.90` of the available/total-memory bound used by the probe |
| Optional batch search | `batch_size='auto'`; increase tested batch sizes until the budget is exceeded |
| Logging | Every 50 updates, plus the first update |
| Evaluation / checkpoint interval | Every 200 updates, plus initial/final evaluation |
| Default validation subset | At most 4 pairs, at most 128 response tokens per pair |
| Run output | `/kaggle/working/flylm2_runs/<timestamp>` |

The notebook checks whether the installed PyTorch build includes `sm_60` when running on a P100. If missing, its bootstrap requests `torch==2.10.0+cu126` and matching torchvision/torchaudio packages from the official CUDA 12.6 index, then rechecks support. A kernel restart is requested only when an incompatible older module is already loaded. These are the notebook's explicit compatibility settings; see the [official PyTorch installation archive][pytorch-install]. Other dependencies include NumPy, pandas, SciPy, PyArrow, tqdm, and SentencePiece, with `sentencepiece==0.2.2` requested when that module is missing.

The fixed batch is tested once; automatic batch selection is a memory-fit search, not a throughput optimum. In the updated notebook, a later cell changes the training batch from 2 to 4 and the update cap from 1,000 to 2,000 after the memory probe. That earlier probe does not establish memory fit for batch 4; rerun it with the intended training batch when validating capacity. If the graph cannot fit, execution stops rather than silently reducing the anatomical graph. Re-executing the training cell initializes a new model. Checkpoints contain model weights and provenance, but omit optimizer state and the data cursor, so exact interrupted-run resumption is not provided.

### Inference and activity playback

The export bundles `model.pt`, `graph/`, `tokenizer.model`, `tokenizer.json`, and its matching `flylm_inference.py` into `flylm2_inference.zip`. Extract it and point `BUNDLE_DIR` in the inference notebook to the directory containing those files. Training data are not required for inference. From the extracted bundle directory, the module interface is:

```python
from pathlib import Path
import flylm_inference

model = flylm_inference.load_model(Path("model.pt"), device="cuda")
answer = flylm_inference.reply(
    model,
    "오늘 기분이 어때요?",
    max_new_tokens=128,
    temperature=0.0,
)
print(answer)
```

Use the module shipped with the checkpoint. `device=None` selects CUDA when available and CPU otherwise; explicit CPU inference is also supported. Temperature zero selects greedily among permitted tokens, while positive temperature samples from their softmax distribution. UNK/BOS/PAD are excluded from generation. An incremental UTF-8 decoder constrains byte-fallback continuation; unfinished trailing fallback bytes are omitted at the generation limit.

Loading checks the architecture identifier, tokenizer hash and special IDs, evidence hash and mode, activity-code hash, body-ID ordering hash, neuron count, and strict parameter shapes. The model should not be paired with a refitted tokenizer or silently reassigned physiological modes.

The inference notebook also records actual model `s` states for offline HTML playback. Its embedded layout covers 165,122 neurons using 140,024 soma positions and 25,098 representative vertices from public skeletons. Skeleton vertices are anatomical display points, not inferred soma positions. Synchronized X–Z and X–Y views show the same recorded activity. Brightness represents filtered model output, not a spike raster or biological recording; 12-fps playback is a presentation clock. MP4 export depends on browser encoding support.

### Outputs and evidential limits

| Artifact | Interpretation |
|---|---|
| `history.json`, `result.json` | Training/validation metrics, configuration, runtime information, and run report |
| `batch_tuning.json` | Tested batch sizes, memory budget, peak reserved VRAM, and probe duration |
| `best.pt`, `latest.pt` | Best observed validation checkpoint and most recent weights |
| `examples.json` | Generated qualitative examples |
| `neuron_ports.csv`, `neurotransmitters.csv` | Anatomical ports and effective transmitter provenance |
| `neuron_activity_annotations.csv`, `activity_types.csv` | Per-neuron assignments and type-level evidence counts |
| `activity_coverage.json`, `dynamics_evidence.json` | Evidence coverage, defaults, version, and transfer assumptions |
| `activity_history.json` | Probe spike events, silent-spiking fraction, mean synaptic signal, and finite-state diagnostics |
| `graph/`, `tokenizer.model`, `tokenizer.json` | Structural and tokenization dependencies required for restoration |
| `flylm2_inference.zip` | Portable inference export |

The numerical checks test threshold events, reset/refractory behavior, zero-input behavior, positive finite surrogate derivatives, exact agreement with the graded recurrence when spikes are disabled, and sparse forward/backward agreement with a small dense reference. These checks establish implementation properties when run successfully; they do not establish language quality or physiological validity.

The default validation subset and the configured update cap are execution checks. Response NLL, perplexity, and token accuracy are reported over the evaluated response tokens. Perplexity and response tokens per second are **SentencePiece-token metrics** and cannot be compared directly with byte-model values. Meaningful comparisons require a fixed held-out dialogue set, matched data exposure, multiple seeds, and separate reporting of equal-token and equal-runtime budgets. No measured throughput, memory-fit result, quality improvement, or mushroom-body benefit is asserted here.

## Research direction: mushroom-body dynamics and plasticity

The next direction is to make the mushroom body already present in MaleCNS functionally more explicit. The [mushroom-body plan](documents/mushroom-body-plan.md), dated 2026-09-09, proposes a staged program: establish neuron/type correspondence, calibrate Kenyon-cell–APL dynamics, validate spatial compartments, and only then introduce local dopamine-dependent plasticity. These are **planned extensions**, not implemented FlyLM-2 capabilities. The plan predates the spiking-default revision and describes the earlier graded baseline. Its future APL assignment remains a literature-supported graded model; the current unmatched-type fallback does not change that physiological motivation.

### Existing anatomical inventory

The checked-in annotation audit reports the following within the retained `Traced` set:

| Population | Left | Right | Total | Selection basis |
|---|---:|---:|---:|---|
| Kenyon cells (KCs) | 2,019 | 2,045 | 4,064 | `class == 'Kenyon_Cell'` |
| Anterior paired lateral neurons (APL) | 1 | 1 | 2 | `type == 'APL'` |
| Mushroom-body output neurons (MBONs) | 48 | 49 | 97 | `class == 'MBON'` |
| PAM/PPL1 dopamine-neuron candidates | 166 | 166 | 332 | DAN class plus the audited PAM/PPL1 names |

APL body IDs are `10977` on the left and `10540` on the right. The 332 DAN candidates are not a fully validated compartment/function assignment. The audited mushroom-body selection has no populated `receptorType` entries. Actual MB edge counts, transmitter signs, and active sensory-to-MB-to-motor paths still require the connection and neurotransmitter audit described in the plan.

KC, APL, MBON, and PAM/PPL1 types are absent from the current activity registry and consequently use the spiking default in mixed mode. This fallback is an engineering assumption, including for APL; it is not evidence that these populations all signal through spikes. Their anatomical inclusion alone does not establish realistic sparse coding or associative learning. The language head currently reads motor neurons, not MBONs.

### Staged implementation

| Stage | Proposed work | Evidence and completion criterion |
|---|---|---|
| **A. Map and audit** | Resolve exact body IDs, types, hemisphere, transmitter provenance, PN→KC, KC→APL, APL→KC, KC→MBON, and recurrent boundary connections; record compartment uncertainty | Match adult circuit descriptions without copying another specimen's counts. Deliver `mb_nodes.csv`, `mb_edges.csv`, `mb_type_evidence.json`, and `mb_compartment_mapping.csv`; validate direction, signs, IDs, and boundaries. [Li et al.][li2020]; [Takemura et al.][takemura2017] |
| **B. KC–APL dynamics** | Register supported KC subtypes, make APL's graded assignment explicit, and calibrate the existing feedback connections using the four-state model | Test stimulus selectivity, adaptation, recovery, and broader KC responses after APL suppression. Calibrate current strength, thresholds, and time constants together. [Turner et al.][turner2008]; [Lin et al.][lin2014]; [Inada et al.][inada2017] |
| **C. Local APL inhibition** | Associate synaptic contacts with coordinates/ROIs and introduce coupled compartment states within the same two APL neurons | Demonstrate spatially dependent inhibition and conserve aggregated synapse counts when splitting contacts. Pair-level counts alone cannot establish local inhibition. [Amin et al.][amin2020] |
| **D. Language contribution** | Compare existing dynamics, KC-only changes, and KC–APL feedback under matched tokenizer, data, exposure, and trainable-parameter scope | Measure held-out language behavior separately from circuit fidelity; test KC/KC→MBON pathway perturbations with matched control perturbations |
| **E. Compartment-specific plasticity** | Add bounded episode-local efficacy changes to existing KC→MBON edges in one verified compartment | Reproduce timing- and compartment-dependent pairing effects before testing language benefits. Preserve supported direct DAN actions. [Hige et al.][hige2015]; [Handler et al.][handler2019]; [Takemura et al.][takemura2017] |

The initial implementation scope is **A + B, followed by the first D comparison**. Spatial modeling depends on contact-level data; plasticity depends on defensible compartment mapping. The original sensory input ports, motor readout, bypasses, and recurrent paths remain available. The mushroom body is not introduced as a mandatory bottleneck for every token.

APL feedback should use the existing anatomical inhibitory path without double-counting it as an additional current. A single state per APL neuron is a spatially averaged approximation. Local modeling requires synapse positions and sufficiently detailed region assignments; a coarse neuropil label may be inadequate. Slow homeostatic adaptation must also be distinguished from per-token adaptation, especially given the conflicting KC/APL adaptations reported by [Bergmann et al. (2026)][bergmann2026].

For the first plasticity experiment, the plan considers the γ1pedc family after confirming its MaleCNS contacts and DAN correspondence. A proposed effective weight is

$$
w_{ij}^{\mathrm{effective}}=w_{ij}^{\mathrm{base}}\,G(p_{ij}^{\mathrm{episode}}),
$$

where $G$ is positive and bounded, starts at one, and preserves the base sign. The episode-local state would depend on recent KC activity and a compartment-specific DAN signal. This is a hypothesis to calibrate against selected pairing experiments, not a universal Drosophila learning law. It is not implemented STDP, reinforcement learning, or a receptor-level molecular model.

### Experimental and state-management requirements

The proposed comparisons are B0 (current FlyLM-2), B1 (KC dynamics), B2 (KC plus calibrated APL feedback), B3 (local APL), and B4 (one-compartment plasticity). APL-pathway suppression, activity-matched top-k, and controlled rewiring are experimental controls. Final candidates should be repeated over at least three seeds, with matched trainable scope and exposure. Top-k and rewiring do not become part of the default anatomical model.

Circuit measurements should include stimulus-window recruitment, population correlations, excitatory/inhibitory currents, latency, adaptation, recovery, and persistent silence. Global silence is not successful sparse coding, and a calcium-derived response fraction is not a per-tick spike fraction. Language measurements should include response NLL/perplexity, length-dependent context retention, repetition, and unseen context–response associations. Acceptable errors and tradeoffs should be fixed after measuring baseline variability, rather than assuming a target such as “5% active KCs” or an unobserved performance improvement.

New compartment or plasticity states must be batch-local, frozen on padding, preserved in value across TBPTT detachment, and reset according to an explicit episode policy. Updates must return new state without hidden side effects so activation-checkpoint recomputation does not apply learning twice. The first plasticity experiment should freeze the corresponding base weights to avoid simultaneous Adam and local updates to the same efficacy. A shared one-dimensional CSR weight array cannot be mutated to represent different plasticity histories for different batch rows; selected-edge correction messages are a proposed implementation strategy.

Inference must not receive target tokens or validation loss as dopamine signals. Serialization must version the state layout, evidence, neuron IDs, compartment mapping, and time interpretation. Required regression checks include old-mode equivalence, batch independence, padding freeze, chunked versus unchunked inference, checkpoint-recomputation consistency, and save/load equivalence. Memory and throughput must be remeasured after state expansion. Proposed names such as `mb_mode`, `kc_apl`, `apl_local`, and `mb_plastic` are not current configuration options.

## References and attribution

The bibliography distinguishes **explicit notebook citations**, **upstream data/prior work**, **methodological references identified from code**, and **future-work evidence**. A code-to-paper correspondence does not establish that the implementation was copied from, directly inspired by, or reproduces every method in that paper. The entries below cover all four research papers explicitly cited by the two FlyLM-2 notebooks, all nine papers in the supplied mushroom-body plan, and the identifiable methodological foundations of the implementation.

### Papers explicitly cited in FlyLM-2

1. **Zhao, A., et al. (2025).** [Eye structure shapes neuron function in Drosophila motion vision][zhao2025]. *Nature*, 646, 135–142. DOI: `10.1038/s41586-025-09276-5`. Used for H2 spiking evidence; the full visual-circuit model is not reproduced.
2. **Gouwens, N. W., and Wilson, R. I. (2009).** [Signal Propagation in Drosophila Central Neurons][gouwens2009]. *Journal of Neuroscience*, 29(19), 6239–6249. DOI: `10.1523/JNEUROSCI.0764-09.2009`. Used for DM1 PN action-potential evidence and the explicitly qualified `DM1_lPN` transfer.
3. **Pokusaeva, V. O., Satapathy, R., Symonova, O., and Joesch, M. (2024).** [Bilateral interactions of optic-flow sensitive neurons coordinate course control in flies][pokusaeva2024]. *Nature Communications*, 15, 8830. DOI: `10.1038/s41467-024-53173-w`. Used for graded HS dynamics; the study's gap-junction mechanisms are outside the current implementation.
4. **Juusola, M., Dau, A., Zheng, L., and Rien, D. (2016).** [Electrophysiological Method for Recording Intracellular Voltage Responses of Drosophila Photoreceptors and Interneurons to Light Stimuli In Vivo][juusola2016]. *Journal of Visualized Experiments*, 112, e54142. DOI: `10.3791/54142`. Used for the graded `R1-R6` assignment.

### Upstream data and documented project precedent

5. **Berg, S., et al. (2026).** [Sexual dimorphism in the complete Drosophila male central nervous system connectome][berg2026]. *Cell*, 189(18), 5504–5526.e15. DOI: `10.1016/j.cell.2026.08.015`. Primary publication for MaleCNS. The earlier [2025 bioRxiv version][berg-preprint] used the title *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system*; these are versions of the same underlying work.
6. **Lappalainen, J. K., et al. (2024).** [Connectome-constrained networks predict neural activity across the fly visual system][lappalainen2024]. *Nature*, 634, 1132–1140. DOI: `10.1038/s41586-024-07939-3`. Explicitly linked in the FlyLM-1 introduction and relevant to the inherited idea of optimizing dynamics under anatomical constraints. It does not demonstrate language modeling with MaleCNS.

### Methodological references identified from the implementation

7. **Kudo, T., and Richardson, J. (2018).** [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing][sentencepiece2018]. *EMNLP System Demonstrations*. arXiv: `1808.06226`. Correspondence: the SentencePiece library and raw-text tokenization pipeline, linked as software in the notebook.
8. **Kudo, T. (2018).** [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates][kudo2018]. *ACL*. arXiv: `1804.10959`. Correspondence: the Unigram segmentation model; stochastic segmentation regularization is not enabled.
9. **Burkitt, A. N. (2006).** [A review of the integrate-and-fire neuron model: I. Homogeneous synaptic input][burkitt2006]. *Biological Cybernetics*, 95, 1–19. DOI: `10.1007/s00422-006-0068-6`. Conceptual background for threshold/reset dynamics, not a source for FlyLM's numerical calibration.
10. **Neftci, E. O., Mostafa, H., and Zenke, F. (2019).** [Surrogate Gradient Learning in Spiking Neural Networks][neftci2019]. *IEEE Signal Processing Magazine*, 36(6), 51–63; published with the subtitle *Bringing the Power of Gradient-Based Optimization to Spiking Neural Networks*. arXiv: `1901.09948`. Correspondence: differentiating through hard spike events using surrogate derivatives.
11. **Zenke, F., and Ganguli, S. (2018).** [SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks][zenke2018]. *Neural Computation*, 30(6), 1514–1541. DOI: `10.1162/neco_a_01086`. Correspondence: the fast-sigmoid voltage-surrogate form; FlyLM does not implement the complete SuperSpike algorithm.
12. **Williams, R. J., and Peng, J. (1990).** [An Efficient Gradient-Based Algorithm for On-Line Training of Recurrent Network Trajectories][williams1990]. *Neural Computation*, 2(4), 490–501. DOI: `10.1162/neco.1990.2.4.490`. Historical context for truncated temporal credit assignment; FlyLM uses fixed chunks and example resets rather than reproducing that paper's online procedure.
13. **Chen, T., Xu, B., Zhang, C., and Guestrin, C. (2016).** [Training Deep Nets with Sublinear Memory Cost][chen2016]. arXiv: `1604.06174`. Correspondence: activation recomputation as a memory/computation tradeoff. The notebook's per-token checkpoint policy does not establish the paper's asymptotic memory bound for this model.
14. **Kingma, D. P., and Ba, J. (2015).** [Adam: A Method for Stochastic Optimization][adam2015]. *ICLR*; preprint first posted in 2014. arXiv: `1412.6980`. Correspondence: `torch.optim.Adam`.
15. **Pascanu, R., Mikolov, T., and Bengio, Y. (2013).** [On the difficulty of training Recurrent Neural Networks][pascanu2013]. *ICML*. arXiv: `1211.5063`. Correspondence: gradient-norm clipping for recurrent optimization; the paper's other proposed regularization is not implemented.
16. **Ba, J. L., Kiros, J. R., and Hinton, G. E. (2016).** [Layer Normalization][ba2016]. arXiv: `1607.06450`. Correspondence: normalization of the motor readout, with the affine parameters disabled in FlyLM.
17. **Press, O., and Wolf, L. (2017).** [Using the Output Embedding to Improve Language Models][press2017]. *EACL*; preprint first posted in 2016. arXiv: `1608.05859`. Correspondence: tying input embeddings to the vocabulary output projection.
18. **Inan, H., Khosravi, K., and Socher, R. (2017).** [Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling][inan2017]. *ICLR*; preprint first posted in 2016. arXiv: `1611.01462`. Independent methodological foundation for weight tying; the additional loss framework is not used.

### Evidence for the proposed mushroom-body program

19. **Turner, G. C., Bazhenov, M., and Laurent, G. (2008).** [Olfactory Representations by Drosophila Mushroom Body Neurons][turner2008]. *Journal of Neurophysiology*, 99, 734–746. DOI: `10.1152/jn.01283.2007`. Motivation for selective KC responses and joint calibration of integration and threshold.
20. **Lin, A. C., Bygrave, A. M., de Calignon, A., Lee, T., and Miesenböck, G. (2014).** [Sparse, decorrelated odor coding in the mushroom body enhances learned odor discrimination][lin2014]. *Nature Neuroscience*, 17, 559–568. DOI: `10.1038/nn.3660`. Motivation for KC–APL feedback and causal suppression controls.
21. **Inada, K., Tsuchimoto, Y., and Kazama, H. (2017).** [Origins of Cell-Type-Specific Olfactory Processing in the Drosophila Mushroom Body Circuit][inada2017]. *Neuron*, 95(2), 357–371.e4. DOI: `10.1016/j.neuron.2017.06.039`. Motivation for KC subtype-specific excitability rather than one universal KC parameter set.
22. **Amin, H., Apostolopoulou, A. A., Suárez-Grimalt, R., Vrontou, E., and Lin, A. C. (2020).** [Localized inhibition in the Drosophila mushroom body][amin2020]. *eLife*, 9, e56954. DOI: `10.7554/eLife.56954`. Motivation for spatially local APL activity and inhibition.
23. **Li, F., et al. (2020).** [The connectome of the adult Drosophila mushroom body provides insights into function][li2020]. *eLife*, 9, e62576. DOI: `10.7554/eLife.62576`. Circuit and compartment reference for KC/MBON/DAN mapping and recurrent pathways. The journal's citation year is 2020; its version of record appeared in February 2021.
24. **Hige, T., Aso, Y., Modi, M. N., Rubin, G. M., and Turner, G. C. (2015).** [Heterosynaptic plasticity underlies aversive olfactory learning in Drosophila][hige2015]. *Neuron*, 88(5), 985–998. DOI: `10.1016/j.neuron.2015.11.003`. Motivation for stimulus-order-dependent, compartment-specific KC→MBON depression.
25. **Handler, A., et al. (2019).** [Distinct Dopamine Receptor Pathways Underlie the Temporal Sensitivity of Associative Learning][handler2019]. *Cell*, 178(1), 60–75.e19. DOI: `10.1016/j.cell.2019.05.040`. Motivation for receptor- and timing-dependent plasticity direction rather than a universal dopamine scalar rule.
26. **Takemura, S.-y., et al. (2017).** [A connectome of a learning and memory center in the adult Drosophila brain][takemura2017]. *eLife*, 6, e26975. DOI: `10.7554/eLife.26975`. Compartment anatomy and evidence for direct DAN effects, including slow MBON depolarization.
27. **Bergmann, G. A., et al. (2026).** [Conflicting adaptations in an inhibitory feedback circuit][bergmann2026]. *The Journal of Physiology*. DOI: `10.1113/JP290394`. Motivation for separating slow homeostatic adaptations from immediate activity regulation; not a prescribed per-token update rule.

### Data and software resources

- [MaleCNS official downloads][malecns-data] — versioned annotations, neurotransmitters, connectivity, synapses, and skeletons. The source project identifies the dataset as CC-BY; retain its attribution when distributing derived data.
- [SentencePiece repository][sentencepiece-code] and [special-symbol documentation][sentencepiece-symbols] — implementation-specific tokenizer and byte-fallback behavior.
- [PyTorch installation archive][pytorch-install] — the source linked by the notebook's GPU compatibility bootstrap.

[berg2026]: https://doi.org/10.1016/j.cell.2026.08.015
[berg-preprint]: https://www.biorxiv.org/content/10.1101/2025.10.09.680999v1
[lappalainen2024]: https://doi.org/10.1038/s41586-024-07939-3
[zhao2025]: https://doi.org/10.1038/s41586-025-09276-5
[gouwens2009]: https://doi.org/10.1523/JNEUROSCI.0764-09.2009
[pokusaeva2024]: https://doi.org/10.1038/s41467-024-53173-w
[juusola2016]: https://doi.org/10.3791/54142
[sentencepiece2018]: https://arxiv.org/abs/1808.06226
[kudo2018]: https://arxiv.org/abs/1804.10959
[burkitt2006]: https://doi.org/10.1007/s00422-006-0068-6
[neftci2019]: https://arxiv.org/abs/1901.09948
[zenke2018]: https://doi.org/10.1162/neco_a_01086
[williams1990]: https://doi.org/10.1162/neco.1990.2.4.490
[chen2016]: https://arxiv.org/abs/1604.06174
[adam2015]: https://arxiv.org/abs/1412.6980
[pascanu2013]: https://arxiv.org/abs/1211.5063
[ba2016]: https://arxiv.org/abs/1607.06450
[press2017]: https://arxiv.org/abs/1608.05859
[inan2017]: https://arxiv.org/abs/1611.01462
[turner2008]: https://doi.org/10.1152/jn.01283.2007
[lin2014]: https://doi.org/10.1038/nn.3660
[inada2017]: https://doi.org/10.1016/j.neuron.2017.06.039
[amin2020]: https://doi.org/10.7554/eLife.56954
[li2020]: https://doi.org/10.7554/eLife.62576
[hige2015]: https://doi.org/10.1016/j.neuron.2015.11.003
[handler2019]: https://doi.org/10.1016/j.cell.2019.05.040
[takemura2017]: https://doi.org/10.7554/eLife.26975
[bergmann2026]: https://doi.org/10.1113/JP290394
[malecns-data]: https://male-cns.janelia.org/download/
[sentencepiece-code]: https://github.com/google/sentencepiece
[sentencepiece-symbols]: https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md
[pytorch-install]: https://pytorch.org/get-started/previous-versions/
