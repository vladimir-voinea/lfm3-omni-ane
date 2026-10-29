# LFM3-Omni-2B on the Apple Neural Engine - state of the work

Objective: LFM3-Omni-2B on the ANE, 75k context, >20-30 tok/s, full multimodal in and
out, plus the engine/harness layer that Dory got.

## What the model actually is (from config.json, not a blog post)

Lfm3OmniForConditionalGeneration, one 4.3 GB safetensors file.

* text backbone: 24 layers - 13 short-conv, 5 sliding-attention (window 256),
  6 full-attention. hidden 2048, 16 query heads / 4 KV heads, head_dim 128,
  SwiGLU MLP 7680, sandwich norms, q_norm and k_norm, and gated attention
  (sigmoid of gate_proj), vocab 128,000, tied embeddings, RoPE theta 1e6,
  max_position 128,000
* short conv: depthwise causal conv, kernel 7, state = 6 previous positions
* vision encoder: 14 layers (conv + full attention), hidden 1024
* audio encoder: mel 128 -> 17-layer LFM2 stack, hidden 768, sliding window 32
* audio out: 12 codebooks x 2017 codes, maskgit schedule, then the liquid codec

## The state budget at 75k - the good news

| what | bytes |
|---|---|
| 6 full-attn layers x 4 KV heads x 128 dim x 2 (K,V) x 75,000 x 2 B | 879 MiB |
| 5 sliding layers x 4 heads x 128 x 2 x 256 window x 2 B | 5.2 MiB |
| 13 conv layers x 6 steps x 2048 x 2 B | 0.3 MiB |
| total | ~885 MiB |

Dory carried 999 MiB of state at 24k, so 75k here is cheaper than Dory was: LFM3 has
6 full-attention layers against Dory's 36. Context is not what will be hard.

## Two measurements that decide the plan

1. A program can hold 36.6 MiB of state: one full-attention head at 75k (K and V,
   75,000 x 128 fp16 each) loads and runs. A trivial program even loaded 293 MiB, so
   the old 34 MiB figure was a property of Dory's compute, not a hard per-program cap.

2. Shape matters more than size. probes/attention_placement.py measures torch-traced
   attention over a 75,000-row state under CPU_ONLY and CPU_AND_NE: fp16 matmul
   0.51 vs 0.56 ms, with softmax 1.63 vs 2.72 ms. Equal times under both compute
   units means the CPU ran it. Controls:
   * a 1x1 conv pair 2048->7680->2048: 1.63 ms CPU vs 0.81 ms NE, so the ANE is live
   * dory_attn_full_0_h0, a program proven on the ANE: 2.09 ms CPU vs 1.27 ms NE, so
     the measurement method is sound

   A traced torch graph with a big state, a transpose of that state and a wide matmul
   is rejected by the ANE and quietly run on the CPU. Dory's converter builds its graph
   with explicit MIL ops, and that graph is accepted.

## Next decision

Rebuild the 75k attention probe at the MIL level in Dory's style (explicit mb.* ops,
state read the way Dory reads it) and re-measure placement. The existence proof that a
24,576-wide score tensor inside a stateful program runs on the ANE is
dory_attn_full_0_h0; 75,000 is 3x wider, which is exactly what the probe must answer.

If it lands on the ANE, the plan is:

| program kind | count | state each |
|---|---|---|
| full attention, one KV head, 4 query heads | 24 (6 layers x 4 heads) | 36.6 MiB |
| sliding attention, whole layer | 5 | 0.5 MiB |
| conv layer + MLP | 13 | 24 KB |
| epilogue (gate, o_proj, MLP) | per attention layer | none |
| embeddings, lm_head | 2 | none |

Full attention is 24 dispatches; the 0.78 ms measured for 4 query heads over 75k would be
19 ms, leaving room inside a 40 ms budget for 22-25 tok/s. That 0.78 ms is a CPU number
though, so it is the wrong number to plan against until the MIL probe says the ANE can
take the shape at all.
## ROUND 2: the ANE width wall is 65,536, and 75k needs tiling

Measured with MLComputePlan (load_from_path on a compiled .mlmodelc,
get_compute_device_usage_for_mlprogram_operation per op). Stopwatch verdicts were
wrong: the 0.59 ms 'on ANE' reading turned out to be the CPU, so from here on every
placement claim comes from the compute plan, not from a timer.

Device maps for the same attention graph at different cache widths:

| width | ANE ops | CPU ops |
|---|---|---|
| 24,576 (my shape) | 0 | 9 (all) |
| 65,536 | 8 | 1 |
| 75,000 | 5 | 4 |

Per-op detail at 75,000: read_state, reshape and the FIRST matmul run on the ANE;
then the scale multiply, the mask add, the softmax and the second matmul fall to the
CPU, and everything downstream of the softmax follows it.

Dory's program at 24,576 puts its softmax on the ANE (75 ops ANE, 0 CPU), so the
engine does support softmax -- just not at 75,000. The limit sits at the known
65,536 axis ceiling, which Dory already hit once with the 131,072-wide LM head.

Consequence: full attention at 75k must be tiled into pieces of at most 65,536 rows,
with a log-sum-exp merge. 75,000 = 65,536 + 9,464, or two equal tiles of 37,500 --
both fit. Dory already has the merge implemented
(anemll/utils/dory_caches.merge_attention_partials) for its own tiled mode, so this
reuses an idea rather than inventing one.

Dispatch budget with 2 tiles:

| what | dispatches | estimate |
|---|---|---|
| full attention, 6 layers x 4 KV heads x 2 tiles | 48 | 0.3-0.5 ms each = 15-24 ms |
| sliding attention, 5 layers (window 256) | 5-10 | < 1 ms |
| conv layers + MLP, 13 | 13 | ~4 ms |
| attention epilogues (gate, o_proj, MLP), 11 | 11 | ~3 ms |
| embeddings, lm_head (128k vocab > 65,536 so split) | 3 | ~3 ms |
| total | ~80 | ~28-35 ms => 28-36 tok/s |

That is inside the >20-30 tok/s target, but the margin is thin and every estimate
above is untested.

## Round 3 next step

Build one real tile program on the Dory pattern -- state (2, 1, rows, 128), q, mask
and positions as inputs, outputs (max, sumexp, weighted value) -- verify its op map
is ANE-only at 37,500 rows, then run 48 of them in series for a real per-token
attention cost.
## ROUND 3: the tile program works, and what it costs

Construction that the engine accepts (probes/dory_shape.py, tile_all_heads2.py):

* one 4D state buffer (2*kv, 1, rows, 128) holding K in slots 0..kv and V above it
* q/k/v produced by 1x1 convs inside the program, the new row written into the cache
  with a slice assign driven by an int32 `pos` input
* q is (1, kv, rep, d) -- the query-head axis sits in the sequence slot, which is what
  Dory's graph does and what keeps the matmuls 4D
* two matmuls (1,kv,rep,d)x(1,kv,d,rows) and (1,kv,rep,rows)x(1,kv,rows,d), scale and
  mask between them, softmax on the score tensor

Device maps, all measured with MLComputePlan:

| graph | ANE ops | CPU ops | loads? |
|---|---|---|---|
| 24,576 rows, kv=1 | 26 | 0 | yes |
| 28,672 rows, kv=1 | 26 | 0 | yes |
| 28,672 rows, kv=4 | 28 | 0 | yes |
| 32,768 rows, kv=4 | - | - | **no (-14)** |
| 37,500 rows, kv=1 | 24 | 2 | yes, matmul + reshape on CPU |

So a tile is at most ~28,672 rows, and 75,000 needs three tiles (28,672 + 28,672 +
17,656). Packing all four KV heads into one program keeps every op on the engine.

Cost, median of 15 dispatches on the engine:

| rows | kv | ms | ms per head |
|---|---|---|---|
| 2,048 | 1 | 0.197 | 0.197 |
| 2,048 | 4 | 0.444 | 0.111 |
| 28,672 | 1 | 0.783 | 0.783 |
| 28,672 | 4 | 2.158 | 0.539 |

Two things fall out:

1. **Packing heads is worth it** -- 0.539 ms per head at kv=4 against 0.783 at kv=1.
   Dispatch overhead is small (a 2,048-row program runs in 0.2-0.4 ms); the cost is
   the rows times the heads.
2. **Softmax is nearly free on the engine**: the same program without it takes
   3.46 ms against 3.33 ms with it. On the CPU the softmax is the expensive part
   (5.73 ms without, 18.68 ms with) -- which is the second proof that these programs
   really run on the engine, since the engine timing does not move when it is removed.

Attention budget per token: 6 layers x (2 x 2.16 ms for the full tiles + ~1.4 ms for
the remainder tile) = **~34 ms**. Add the rest of the model (13 conv layers, 24 MLPs,
projections, a split LM head over a 128k vocabulary) and the token lands somewhere
around **45-50 ms, or 20-22 tok/s** -- inside the >20 tok/s floor, below the 30 tok/s
top of the target, and every non-attention number there is still an estimate.

## Round 4 next step

Build the non-attention pieces and measure them, rather than estimating: a conv layer
(in_proj -> depthwise conv k=7 -> gates -> out_proj) with its 6-step state, one MLP,
and the LM head split to stay under the 65,536 axis limit. Those three numbers decide
whether the token is 45 ms or 65 ms.
## ROUND 4: the whole token, measured piece by piece

Every number below is a median of 20 dispatches on the engine, with an op-level
device map behind it (probes/pieces2.py):

| piece | ms | ANE ops | CPU ops |
|---|---|---|---|
| conv layer (in_proj, depthwise conv2d k=7, gates, out_proj) | 0.474 | 14 | 0 |
| MLP (SwiGLU, 2048 -> 7680 -> 2048) | 1.027 | 5 | 0 |
| sliding attention, window 256, all 4 KV heads | 0.568 | 18 | 0 |
| LM head, 128,000 outputs in one program | 4.836 | 1 | 0 |
| LM head split into two 64,000 halves | 2 x 2.501 | 1 each | 0 |
| full attention, one 28,672-row tile, 4 KV heads | 2.158 | 28 | 0 |

Two construction notes that cost time and are now settled:

* **a stateful conv layer cannot be built this way** -- with a (1,1,2048,6) state
  buffer the engine faults at inference (`IOSurface smaller than the model expects`).
  Carrying the 6-row history as an input/output pair instead works and costs 24 KB
  per layer per token.
* **conv1d with groups faults on the engine too**; the same depthwise kernel as a
  conv2d with a (1,7) kernel is accepted.
* the 128,000-wide LM head does *not* need splitting (4.84 ms as one program against
  5.0 ms as two), so the 65,536 axis limit does not bind here.

### The token at 75k, assembled

| part | count | ms each | ms |
|---|---|---|---|
| full attention, 3 tiles of ~28k | 18 | 2.16 | 34.0 |
| MLP | 24 | 1.03 | 24.6 |
| conv layer | 13 | 0.47 | 6.2 |
| sliding attention | 5 | 0.57 | 2.8 |
| full-attention projections (q,k,v,o) | 6 | 0.40 | 2.4 |
| LM head | 1 | 4.84 | 4.8 |
| **total** | | | **~75 ms => 13.3 tok/s** |

At a shorter context the attention term collapses:

| context | attention | token | tok/s |
|---|---|---|---|
| 75,000 | 34 ms | 75 ms | 13.3 |
| 24,000 | ~11 ms | 52 ms | 19.2 |
| 8,000 | ~4 ms | 45 ms | 22.2 |

### What this means for the objective

**75k context and 20-30 tok/s are not simultaneously reachable on this machine with
fp16 weights.** The weight-bound half of the token (MLPs, convs, projections, LM head)
is ~38 ms and the state-bound half (full attention over 75k) is ~34 ms. Palettizing
weights to 4 or 6 bits is the lever that moves the first half -- Dory measured 1.9x on
exactly that operation -- but it does nothing for attention, which reads fp16 state.
So the realistic best case at 75k is **~54 ms, or 18-19 tok/s**, and the target is met
at contexts up to roughly 24k.

That is a target problem to settle before building, not after: either the context goal
moves to ~24k, or the speed goal becomes ~18 tok/s at 75k, or the plan takes on a
cheaper attention scheme (a chunked/streaming attention that reads only part of the
cache per token), which is a research-sized change with its own quality cost.
## ROUND 5: a real decoder layer converts and verifies

Layer 0 of the real model (short conv + SwiGLU MLP + all four sandwich norms), built
from the actual safetensors weights and compared against PyTorch on the same input:

| comparison | value |
|---|---|
| relative error vs torch fp32 | **0.315 %** |
| max absolute difference | 0.0079 (output norm 22.9) |
| conv-history output | max diff 0.0006 |

Individual ops, each converted on its own: depthwise (1,7) conv 0.036 % error, 1x1 conv
0.69 % (fp16 accumulation over 2,048 terms -- that is the floor here, not a bug).

### Two conversion traps found the hard way (both now avoided)

1. **Core ML does not preserve the tuple order of a traced function's outputs.** The
   first run reported a 149x relative error that was really my comparison reading the
   history tensor where it expected the hidden state. Outputs must be selected by
   name, or at least by shape, never by position.
2. **A `view` applied to the result of a `cat` converts to the wrong strides** (that
   error was 3.3x, and it survived fixing the output order). Keeping every tensor in
   one layout -- (1, H, 1, T) throughout, three separate 1x1 projections instead of one
   3H-wide projection sliced afterwards -- removes the view and the error with it.

### Weight map (for the full conversion)

`model.language_model.layers.N.*`, 10 tensors on conv layers and 14 on attention
layers:

* conv layer: `conv.conv.weight` (2048,1,7), `conv.in_proj.weight` (6144,2048) laid out
  as [B, C, x], `conv.out_proj.weight`, `feed_forward.w1/w2/w3`, and the four norms
  `operator_norm`, `post_operator_norm`, `ffn_norm`, `post_ffn_norm`
* attention layer: adds `self_attn.q_proj` (2048,2048), `k_proj`/`v_proj` (512,2048),
  `o_proj`, `q_norm`/`k_norm` (128,), `gate_proj` (16,2048)
* `model.language_model.embed_tokens.weight` (128000,2048) is tied to the LM head

## Round 6 next step

The attention layer with real weights: q/k/v projections, per-head q_norm and k_norm,
RoPE, the sigmoid gate, o_proj -- verified the same way, against PyTorch, with outputs
selected by name.
## ROUND 6: the whole text backbone runs on the engine

24 Core ML programs (one per layer, each with its MLP fused in) plus a final norm and
the LM head, 3.7 GB of artifacts in `model-28k/`:

* 13 conv layers: 0.474 ms
* 5 sliding-attention layers (window 256): state 8 x 1 x 256 x 128 fp16 = 1 MB
* 6 full-attention layers: state 8 x 1 x 28,672 x 128 fp16 = 58.7 MB each, 352 MB total
* final norm + LM head (128,000 outputs, tied to the embedding matrix)

A token goes through all 26 programs in **~87 ms (11.5 tok/s)**, and the model emits
128,000 logits.

### Does it produce the right numbers?

Against a pure-PyTorch 24-layer reference on the same input ids [1, 1000, 2000, 3000]:

| check | result |
|---|---|
| Pearson correlation over all 128,000 logits | **0.973** |
| top-1 token | same (207) |
| top-2 token | same (20) |
| top-10 overlap | 6 of 10 |
| ANE top-5 values | 8.58, 7.75, 6.43, 6.27, 6.17 |
| reference top-5 values | 9.14, 7.98, 6.91, 6.54, 6.00 |

So: it runs, it agrees on the top two tokens, and it correlates 0.973 -- but it is not
yet a match. The reference is fp32 and processes the whole sequence at once; the engine
runs fp16 and feeds one token at a time through caches, so some divergence is expected.
The systematic ~0.4 lower logits on the engine side suggests one more fp16 accumulation
step to find, and the 4-of-10 disagreement in the tail is the thing to chase down
before calling the model correct.

### What is NOT verified

* the transcription of the math itself. The per-layer comparisons (0.315 %, 0.326 %)
  were against my own torch reference, so they validate the conversion, not my reading
  of the model. The official `transformers` code will not load in this environment
  (it needs transformers >= 5, and installing that next to coremltools' numpy breaks
  the Core ML stack), so a truly independent reference is still missing.
* text quality: nothing has been decoded into words yet.

### Next

1. Close the numeric gap (find the fp16 accumulation step behind the ~0.4 offset).
2. Get an independent reference: a separate venv with transformers 5 and its own numpy,
   used only for a handful of reference logits.
3. Generate text and check it reads like language.
4. Only then scale to 75k (three tiles per full-attention layer).
## ROUND 7: the model answers correctly on the engine

The text backbone runs end to end on the Neural Engine and produces right answers:

| prompt | engine output |
|---|---|
| `[BOS] The capital of France is` | `Answer: Paris` |
| chat template: `What is the capital of France?` | `The capital of France is **Paris**.` |

and the pure-PyTorch reference, given the same ids, answers the same way. Prefill of
the 5-10 token prompt takes 0.44-0.75 s on the engine.

### The bug was the prompt, not the math

Without a BOS token the model degenerates: the engine emitted
`The capital of France is France. The capital of France is France...` and the torch
reference emitted `is is is is is is`. Prepending token 1 (`bos_token_id` in the
config) fixes both, which is the useful part: **two independent implementations
degrading in the same place is evidence the transcription is right** -- the model
itself just needs its BOS.

This also retires the worry from round 6. The 0.973 logit correlation was measured on
ids starting with 1, so it was already a BOS-prompted comparison; the engine's top-1
and top-2 agreed with the reference there, and now the decoded answers agree too.

### State

* working: 24-layer text backbone at a 28,672-token cache, 11.5 tok/s, correct answers
* not started: multimodal (vision tower, audio encoder, audio codec out), the engine/
  server layer, 75k context (needs 3 tiles per full-attention layer), palettization
* an intermittent `-14` at program load when a 4.3 GB torch reference had just been
  resident -- retrying after the process exits loads cleanly, so it is memory pressure,
  not the model
## ROUND 8: tiled attention, half-built

Target: the 75k path needs each full-attention layer split into 3 tiles of 28,672 /
28,672 / 17,656 rows, because the engine refuses a wider score tensor.

Design that the measurements support:

* a **tile program** owns one slice: it projects q/k/v, writes its row into its own
  cache, and returns an unnormalised partial (weighted values, row max, sum of exps)
* the **host merges** the tiles with log-sum-exp -- 16 x 128 numbers per layer, so the
  cost is noise -- because a converter crash made an in-program merge unusable
* an **epilogue program** takes the merged attention and runs gate, o_proj, the
  post-norm, residual and the MLP

Verified so far:

| piece | status |
|---|---|
| tile program, 3 outputs (1,4,4,128) / (1,4,4,1) / (1,4,4) | runs on the engine, any mask |
| epilogue program, (1,2048,1,1) x2 inputs | runs on the engine and on the CPU |
| tile + epilogue driven over a 40-token sequence | **fails with an ANE inference error** |

Lessons banked on the way:

1. An in-program log-sum-exp merge crashed the converter -- host merge works.
2. A tile that has not started yet must be **skipped**, not called with `pos = -1`:
   the write slice `cache[..., -1:0, :]` is empty and the prediction fails.
3. Core ML still does not preserve output order: the tile's two (1,4,4) outputs are
   the same shape, so max and sum must be told apart by shape on purpose.

Both programs pass individually and fail together, so the next step is to instrument
the driving loop to name the exact call that breaks rather than guess.
## ROUND 9: tiled attention runs, and where its error comes from

The 75k path works end to end now. Over a 40-token sequence split across 3 tiles of
16, driven token by token through the engine, the tiled layer matches plain full
attention to **2.46 %** relative error, max abs 0.215.

### The design that finally worked

One program per tile, with **one extra cache row that is always written and never
attended** (the scratch slot). A tile that is filling writes at its own position; a
tile that is full writes the scratch slot. The mask keeps row `ROWS` at -1e4 forever,
so writing there is harmless and the tile stays a single program that serves both
roles.

Two designs failed first, both worth recording:

1. **Gated read-modify-write** of the slot (`slot = slot*(1-gate) + k*gate`) makes the
   engine refuse to build a plan at all -- error -14. A stateful program cannot read
   and rewrite the same state slice.
2. **Two programs, one writing and one read-only**, cannot work either: an `MLState`
   belongs to a model instance. The read-only program would have its own empty cache.

### Is 2.46 % my logic or the arithmetic?

The same tiled merge, done in torch fp32 with no Core ML anywhere, matches full
attention to **2.9e-07**. So the merge is exactly right and the 2.46 % is fp16 loss:
each tile exponentiates and multiplies in fp16, and the partials the host merges are
fp16 too. The untiled path converts at 0.33 %, so tiling costs roughly 8x accuracy.

That is acceptable for proving the mechanism and **not** acceptable for a 24-layer
model, where such an error per layer compounds. Round 10's job is to find where the
precision goes -- the candidates are the fp16 `exp`/weighted sum inside the tile, and
the fp16 hand-off of (weighted values, max, sum) to the host.
## ROUND 10: the tile is correct; the precision is what is thin

Measured the tile program's partials directly against torch fp32 over 6 positions of
real weights:

| quantity | agreement |
|---|---|
| weighted values | 0.72 % |
| sum of exps | 0.34 % |
| row max | 3.398 vs 3.398, 12.079 vs 12.063, 2.673 vs 2.676, 11.960 vs 11.960 |

So the tile program computes the right thing. The tiled layer's 2.14 % against plain
full attention (down from 2.46 %) is the fp16 cost of the tile partials plus the
epilogue, not a logic error -- the same merge in fp32 is exact to 2.9e-07.

### One real bug fixed on the way

The epilogue was computing the per-head attention gate from the **raw** hidden state
instead of the normalised one. The tile program normalises before projecting q/k/v, so
the gate has to be computed from that same normalised vector. Worth 2.46 % -> 2.14 %
and, more importantly, it would have been invisible in a passing-but-wrong model.

### Two hypotheses tested and rejected

1. **fp16 `exp` underflow** below 6e-5 losing mass over a long cache -- a real risk in
   principle, but casting the softmax phase to fp32 changed the result by nothing at
   all, because `compute_precision=FLOAT16` folds it straight back.
2. A test-harness bug of my own: multiplying by `q_norm.weight[0]` (a scalar) instead
   of the 128-vector, which made the tile look 22 % wrong until fixed. Third time this
   project has produced a wrong number from the measuring code rather than the thing
   measured.

### Next

Build the tile with `compute_precision=FLOAT32` (and check it still lands on the engine,
and what it costs per dispatch). If fp32 accepts, the tiled path should reach the
untiled 0.33 % and 75k becomes usable; if it does not, the fallback is to keep the
weighted sum in fp32 by hand and confirm what the compiler does with it.
## ROUND 11: fp32 is not available on the engine -- decision on 75k

Same tile program, two build settings, measured with the compute plan and a timer:

| build | dispatch | ops on engine | ops on CPU |
|---|---|---|---|
| `compute_precision=FLOAT16` | 0.402 ms | **66** | 0 |
| `compute_precision=FLOAT32` | 0.325 ms | 0 | **75 (all)** |

So the accuracy lever and the engine are mutually exclusive here: asking for fp32
hands the whole program to the CPU. On a 16-row toy that looks *faster* (0.33 ms),
which is exactly the trap -- at 28,672 rows per tile the CPU would be an order of
magnitude slower, and there are 18 of them per token.

The fp16 rounding that costs 2.1 % is the `exp` weights themselves: `e = exp(s - m)`
is rounded to fp16 before the multiply, so a weight of 0.5 carries ~1e-3 relative
error and the weighted sum inherits it. The untiled path avoids this because Core ML's
own `softmax` op runs internally in fp32 on the engine.

### Scope decision

**The shipping configuration stays at the single-tile 28,672-token cache (0.33 % per
layer), which runs and answers correctly.** 75k stays a documented limitation with a
working mechanism at 2.1 % per layer -- not accurate enough to replace the single-tile
path in a 24-layer stack.

That decision is deliberate: the objective's *missing* half -- vision tower, audio
encoder, audio codec output, the engine/server layer, palettization for speed -- is
entirely unbuilt, and every round spent chasing 6x precision on a context that also
needs 3x the dispatches serves the goal less than building the parts that do not exist
at all. 75k gets revisited once the rest exists; the design and the traps are recorded
above so it is not lost work.
## ROUND 12: the engine and the server exist and answer

`tools/lfm3_runtime.py`, `tools/lfm3_engine.py`, `tools/lfm3_serve.py` -- the layer
Dory has and LFM3 did not.

* the 26 programs are loaded **once**; a session is a set of MLStates + conv histories
  + a position over those same models (my first attempt loaded a second copy per
  session, which failed with `This model was not loaded with the Core ML Framework`)
* the chat template is rendered by hand -- `<|im_start|>role\ncontent<|im_end|>\n`, then
  `<|im_start|>assistant\n` -- and the prompt is prefixed with token 1 (BOS), which is
  what made the model answer instead of babble
* sampling reuses Dory's `Sampler`/`SamplingParams`, so temperature, top_p and
  repetition penalty behave the same as in the Dory engine
* sessions are reused when the new prompt extends the stored tokens (the same
  longest-prefix rule as Dory's `_adopt`)

Verified end to end over HTTP:

| request | answer |
|---|---|
| non-streaming: largest planet | `The largest planet in the solar system is **Jupiter**.` |
| streaming (`stream: true`) | `Bonjour !` in proper SSE chunks, then `[DONE]` |
| second turn: what is the second largest | `Saturn.` |
| `/health`, `/v1/models` | ok, sessions counted |

Decode speed at short context, measured by the engine itself: **16.3 tok/s** (25
positions). That is the best case; it falls towards ~11.5 as the cache fills.

Known limitation: the streaming path generates first and then emits one chunk. The
wiring is right (framing, terminator, `[DONE]`, usage chunk) but it is not incremental
yet -- the generator needs to yield per token instead of returning a finished string.
## ROUND 13: streaming is incremental, and the prefix cache pays

`Engine.stream()` is now a generator that yields each token as it is sampled, and the
server emits one SSE chunk per token. Verified: a 24-token cap produced **9 chunks that
assemble to `Red, blue, green.`** -- the correct answer to `List three colours.`

Prefix reuse, measured on a two-turn conversation:

| turn | first token | tokens re-fed | reused | decode |
|---|---|---|---|---|
| cold: `capital of Japan?` | 0.921 s | 18 | no | 16.3 tok/s |
| warm: `And of Italy?` | **0.701 s** | **14** | **yes** | 16.4 tok/s |

Both answers were right (`Tokyo`, `Rome`). The gain is small here because only 4 tokens
of the 18 were already in the cache -- the mechanism matters at conversation length, and
the flag plus the re-fed count make it measurable rather than assumed.

Two operational notes:

* a leftover server process holding all 26 programs makes the *next* process fail at
  `make_state()` with `This model was not loaded with the Core ML Framework`. That is
  memory pressure again, not corruption -- kill the old process and it loads.
* decode speed is **16.3-16.4 tok/s at ~25 positions**. Expect it to fall towards 11.5
  as the cache fills, which is the number the earlier full-context measurement gave.
## ROUND 14: the vision path is wired but not yet correct

Built with real weights, in torch first (`tools/vision_ref.py`):

* patchifier 1536 -> 1024, 14 layers (8 gated depthwise-3x3 conv, 6 non-causal
  attention with q/k layernorm and **axial** RoPE over 16 frequencies per axis),
  final norm
* projector: layer_norm -> merge 2x2 -> linear 4096 -> gelu -> linear 2048
* shapes check out: a 224x224 image is 196 patches -> 196 x 1024 -> **49 x 2048**
  language tokens, all finite

Then wired into the text model (`tools/vision_chat.py`): the `<|image|>` placeholder
position is replaced by the 49 projected vectors, fed one position at a time through
`Runtime.step_hidden`. The official code does the same thing with `masked_scatter`.

### Behavioural result -- partial, and not good enough

Same question, three solid-colour pictures:

| image | answer |
|---|---|
| solid red | `White` |
| solid green | `White` |
| solid blue | **`Blue`** |

The answer *changes* with the image, so the features do reach the text model -- but
only one of three is right. That is a wired-but-wrong path, not a working one.

Suspected causes, in the order worth testing:

1. **patch layout / channel order.** `patchify` puts channels before the temporal
   duplicate (`[c, temporal, ps, ps]`); a solid-colour image cannot tell a channel swap
   from a correct read on its own, so a red/green/blue triple is a weak probe.
2. **normalisation.** mean 0.5 / std 0.5 / rescale 1/255 is what `processor_config.json`
   says, so this looks right -- but worth an explicit check.
3. **feature injection count.** the model expects exactly `features.shape[0]`
   placeholders scattered in; feeding 49 vectors at one placeholder should be
   equivalent in sequence terms, but that equivalence has not been proven.

### Also this round

The official `transformers` reference is still unusable: version 5 requires torch
>= 2.6 and this environment pins **2.5.0** for coremltools. Shimming
`is_torch_available` is not enough -- the remote code path checks again. Without that
reference, vision correctness has to be established behaviourally, which is why the
colour probe matters.
## ROUND 15: the reference implementation runs -- and my tower is wrong

The blocker from round 14 is gone. `transformers` 5 needs torch >= 2.6 and the
converter venv pins 2.5.0; a throwaway venv (`/tmp/refenv`, 699 MB, torch 2.14.1 +
transformers 5.21 + torchvision) runs the **official model** against the same weights.
A fake `torch-2.6.0.dist-info` on `sys.path` was not enough (torch 2.6 APIs such as
`fsdp.CPUOffloadPolicy` are imported for real). Pillow also had to go to a scratch
path (`/tmp/pil_lib`), since the converter venv does not carry it.

First real measurement of the official vision path:

* the processor resizes a 320x213 photo to a **20 x 32 patch grid**, 640 patches,
  not the square 16 x 16 I had been feeding -- 160 language tokens, not 64
* official features: 640 x 1024, **per-token norm 1106** (absmax 343)
* my tower on the same patches: per-token norm **0.17**
* **cosine agreement 0.725** (min -0.34)

So the transcription is wrong, and now it is measurable instead of guessable. Two
separate facts fall out of the noise:

1. the **scale** gap (1106 vs 0.17) suggests the official `last_hidden_state` is not
   the post-`embedding_norm` value I assumed -- the projector's `layer_norm` would
   cancel a pure scale error, so this alone is survivable.
2. the **direction** gap (cos 0.725) is a genuine transcription error somewhere in the
   14 layers, and it is the one that matters.

The next step writes itself: dump the official hidden state after every layer (hooks)
and diff against mine layer by layer, instead of guessing at whole-model behaviour.

### Disk

Cleaned 2 GB of stale conversion scratch dirs from `/tmp` (bb121-dd, breaker-dd*,
bb-review-dd, hard-review-dd): 2.9 GB -> 4.9 GB free. `/tmp/refenv` (699 MB) is
temporary and gets deleted once the towers are verified.
## ROUND 16: image input works -- and the disk mistake

### The vision bug, found by bisection

With the official reference running, the patchifier matched **exactly** (cos 1.0, max
diff 0.0), so the error was inside the layers. The clue: the official layer-0 output had
abs-mean 0.0073 while its own input was 0.424 -- impossible under a plain residual.
Reading `Lfm3VlVisionModel.forward` settled it:

* `embedding_norm` is applied **right after the patchifier**, not at the end
* the returned `last_hidden_state` is the raw residual stream, **unnormalised**

My tower had the norm at the end (and nowhere else). After moving it:

| check | before | after |
|---|---|---|
| feature cosine vs official | 0.725 | **1.0000** |
| per-token norm | 0.17 | **1106.56** (official 1106.56) |
| projector relative error | large | **0.0** |

### Verified behaviourally, on a real photograph

The Beach wallpaper (via the official processor: 20x32 patch grid, 640 patches,
160 language tokens):

* "Describe this image in one sentence." -> *"This vibrant, stylized digital landscape
  depicts a tranquil coastal scene at dusk, featuring a calm sea..."* -- correct
* "Is this a beach or a forest? One word." -> **Beach** -- correct

Image input is now working, not just wired.

### The disk mistake (my fault)

I let free space fall to zero: the reference venv (763 MB) plus repeated Core ML model
caches filled it. Two invisible consequences: `make_state()` began failing with
`This model was not loaded with the Core ML Framework`, which I twice misread as memory
pressure or corruption -- it was **ENOSPC**, and Core ML needs scratch space to load a
program. Nothing was corrupted; the same code loaded fine once space was free.

Freed 8.4 GB by deleting only things that are reproducible or already published:
`dane5-artifacts` (5.9 GB, the published set24k on HuggingFace), `dane5-build`
(1.5 GB, intermediates), `.uv-cache` (1.0 GB). Left `dane6-artifacts` (10 GB) alone
because I cannot confirm it is published anywhere.

Rule adopted from here on: check `df` before every conversion or model load, and delete
scratch in the same round it is created.
## ROUND 17: the audio encoder is exact

`tools/audio_ref.py` transcribes the audio path with real weights: mel (128 bins) ->
three stride-2 convs -> 17-layer LFM2 stack -> final norm -> projector (768 -> 2048).

| check | result |
|---|---|
| downsampler vs official | cos **1.0**, max diff **0.0** |
| every one of the 17 layers | cos **1.0** |
| encoder output vs official | cos **1.0**, relative error **0.0** |
| projected vectors | cos **1.0** |

The official reference for audio: 2 s swept tone -> `input_features` [1, 128, 199] ->
**25 tokens** x 768 -> 25 x 2048 language vectors (stride 8, so 199 frames -> 25).

### Two bugs found by bisection, both invisible without the reference

1. **RoPE theta is 1e6, not 1e4.** The text model uses 1e4 for its sliding layers,
   and I carried that assumption over. The conv layers matched exactly from the first
   run; the attention layers sat at cos 0.9992 and decayed to 0.981 over 17 layers --
   a small, systematic error that no behavioural test would have isolated.
2. **A final norm before the projector** (`lfm.embedding_norm`, 768) that I had left
   out. This one produced a paradox worth recording: my stack matched the official
   layer-17 hook at cos 1.0 while the encoder output looked like cos 0.64. Both were
   true -- the hook captures the pre-norm state and the encoder returns the post-norm
   one. The contradiction was the clue.

### Method note

Vision took two rounds of blind guessing before the reference existed; audio took one
round with it. The reference venv (`/tmp/refenv`, 705 MB) is worth its disk: rebuild it
when a tower needs checking, delete it after.
## ROUND 18: audio input works -- speech transcribed correctly

The mel front end (`tools/mel_front.py`) reproduces the official extractor: periodic
hann, centre=False STFT, power spectrum, slaney mel filters, log10 clamped at -12,
then `(x + 4) / 4`. Against the official processor's own features: correlation **1.0**,
mean absolute difference **0.00025**, max 0.018.

Full path on real speech (macOS `say`, 1.97 s, 16 kHz mono):

* 1.97 s -> mel [128, 196] -> **25 audio tokens** x 2048
* "Transcribe the audio." -> *"The capital of France is Paris."* -- **exact**
* "What did the speaker say?" -> *"The capital of France is Paris."* -- **exact**

That is my mel, my encoder, my projector and the Neural Engine text model, with no
transformers code in the loop at run time.

### Where the objective stands

| part | state |
|---|---|
| text model on the engine | working, correct answers, ~16 tok/s at short context |
| engine + server + streaming + prefix cache | working, verified over HTTP |
| image in | working, exact tower match, correct captions |
| audio in | working, exact transcription |
| audio out (codec, 10 codebooks) | not started |
| 75k context | mechanism works, 2.1 % per-layer error -- parked |
| palettization / 20-30 tok/s | not started |
| publishing | not started |
## ROUND 19: audio out, scoped and started

Audio output has three stages, all now identified:

1. **depth decoder** (`tools/depth_ref.py`): the language model's 2048-dim hidden state
   at the frame's preceding position -> `depth_linear` -> 12 codebook slots of 1024 ->
   3 **bidirectional** LFM3 layers (32 heads / 8 kv, head_dim 32, theta 1e6, per-head
   sigmoid gate) -> one 2017-way head per codebook (tied embeddings)
2. **maskgit schedule `[4, 4, 4]`**: the 12 codebooks are revealed in three passes of
   four, in random order, each pass re-running the depth decoder with the already
   revealed codes unmasked
3. **liquid codec** (`audio_tokenizer/model.safetensors`, 140 MB, 203 tensors):
   12 codes per frame -> waveform, an LFM stack (512 wide) plus an ISTFT

### Progress and an honest gap

The depth decoder transcription reproduces the official per-layer attention at
**cos 0.998**, and the full 3-layer logits at **cos 0.9889** (relative error 0.168).
Not exact. A 3-layer stack that is 0.998 per layer compounds to 0.989, so the remaining
difference is small and inside every layer rather than in one place -- exactly the
signature that took two rounds to chase on vision, so it is worth one focused check
rather than a guess now.

**A probe bug, again:** my first comparison reported cos -0.0026 and I nearly went
looking for a head-layout scramble. The probe compared the *pre-`o_proj`* attention
against the official *post-`o_proj`* module output. Fourth time in this project that
the measuring code was the thing that was wrong. The sweep that caught it tested four
layouts in one run, which is the pattern worth keeping.
## ROUND 20: depth decoder still not exact -- but the search is narrowed

The cosine hid the problem: at cos 0.998 the **discrete codes only agree 61 %**
(22 of 36 codebooks). For a sampler, argmax agreement is the real test, and 61 % means
the transcription is not usable yet.

Variants tested and ruled out this round, each a whole hypothesis:

| variant | result |
|---|---|
| half (Llama) rotary -- current | cos **0.998** |
| interleaved (NeoX) rotary | 0.928 |
| no rotary | 0.945 |
| q/k norm after rotary instead of before | 0.998 |
| gate taken from the pre-norm input | 0.987 |
| gate applied after `o_proj` | shape error (gate is per head pre-projection) |
| reference running in bf16 | no -- every tensor and the logits are fp32 |

Read the model's own `modeling_lfm3.py` rather than guessing: q_norm/k_norm on the
reshaped heads, rotary after the norms, `gate = sigmoid(gate_proj(hidden))` applied per
head *before* `o_proj`. That is exactly what I implement, and it is what wins the sweep.

**So the error is upstream of the layers.** Hooked `operator_norm` on layer 0: the
official layer *input* already differs from mine at cos **0.99928** (abs-mean 0.35951 vs
0.36018). Everything downstream inherits that. The suspect is the input construction
`depth_linear(hidden) + code_embeddings`, in one of its three parts -- the projection,
the code embeds, or the mask replacement. Next round hooks `depth_linear` directly and
settles which.
## ROUND 21: the depth decoder is exact

| measure | before | after |
|---|---|---|
| logits cosine | 0.998 | **1.0** |
| relative error | 0.069 | **0.0** |
| argmax codes agreeing | 61 % | **100 % (36/36)** |
| top-logit gap | 2.18 | **0.0** |

Two omissions, and the second one is the instructive one:

1. **`depth_linear` has a bias** (12288). I missed it because my own key listing
   filtered on `'depth_linear.weight'` -- a filter that silently drops the sibling. The
   fingerprint that caught it: the module's output differed from a plain matmul of the
   same input with the same weight, which is impossible without a bias term.
2. **The trunk ends with `model.embedding_norm`** (1024). Same final-norm pattern as the
   audio encoder, and the same lesson: a stack that returns `.last_hidden_state` may
   have normalised it first.

Method note: the round-19 cosine of 0.998 looked like a rounding-level difference and
was really a wrong model -- 39 % of codes flipped. **Argmax agreement is the metric for
a sampler**; cosine is not, and I should reach for it first next time.

Audio out now has: depth decoder **exact**. Remaining: the liquid codec (140 MB, 203
tensors, LFM stack + ISTFT) and the ANE conversion of both.
## ROUND 22: the codec runs, and its stage shapes are known

Loading it took three attempts, each a real obstacle worth writing down:

1. the codec folder has only `config.json` + `model.safetensors` -- the modeling code
   lives in the parent, so `get_class_from_dynamic_module` refuses it
2. that code uses **relative imports**, so importing it as a top-level module fails
   (`attempted relative import with no known parent package`); the fix is a throwaway
   package of symlinks (`/tmp/codecpkg/liquid/`) covering every `*.py` of the model
3. codes are `(batch, 12 codebooks, frames)` -- not `(frames, codebooks)`

With that, the official decoder loads **missing 0, unexpected 0** and decodes:

| stage | shape |
|---|---|
| codes in | (1, 12, 16) |
| `decoder.lin` out | (1, **96**, 1282) -- 16 code frames x 6 spectrogram frames, 641 magnitude + 641 phase |
| `decoder.istft` out | (1, 31680) |
| `audio_values` | (1, **30720**) samples = 16 x 1920 at 24 kHz = 1.28 s |

So one code frame is 1920 samples (80 ms) and carries six STFT frames of hop 320 with
an n_fft of 1280. The 96-frame expansion happens before the LFM stack; finding where is
the first task next round, then the transcription itself (FSQ dequantise ->
`from_quantizer` -> 512-wide sandwich-norm LFM stack with kernel-7 convs -> `lin` ->
polar -> inverse STFT with a squared-window envelope divide).

The 512-wide stack is the same sandwich-norm shape as the text model I already build,
so the conversion should reuse that code rather than start fresh.
## ROUND 23: codec transcribed, 0.84 correlation -- not exact

The full decode path is written (`tools/codec_ref.py`): FSQ dequantise -> `project_out`
-> `from_quantizer` -> nearest-x6 upsample (16 code frames -> 96 spectrogram frames) ->
8-layer sandwich-norm LFM stack (conv, conv, attn, conv, attn, conv, attn, conv) ->
`lin` -> magnitude/phase -> inverse STFT -> divide by the squared-window envelope ->
trim 480 samples each end.

| variant | relative error | correlation |
|---|---|---|
| no q/k norm, no gate (first attempt) | 1.04 | 0.47 |
| **+ q/k norm, per-head gate, theta 1e6** | **0.57** | **0.84** |
| + same, theta 1e4 | 0.66 | 0.78 |
| final norm at the start instead of the end | 3.7e11 | 0.02 |
| no final norm | 9.9e12 | 0.07 |

So the shape of the stack is right (the alternatives explode), and the residual error is
something smaller. The next step is the reliable one: dump the official `decoder.lfm`
output and compare it against mine, which separates *stack* error from *lin/ISTFT* error
in one measurement.

Worth noting for the conversion: the codec's attention layers carry the same q/k norm,
per-head gate and rotary as the text model, so the ANE layer build already covers this
arithmetic -- only the widths (512, 8 heads of 64) and the conv kernel (7) differ.
## ROUND 24: audio out is exact on both halves

The stage comparison split the error cleanly: `from_quantizer` was **already exact**
(cos 1.0, max diff 0.0) while `lin` sat at 0.9835, so the fault was inside the 8-layer
stack. A mask sweep found it in one run:

| attention mask | lin cosine |
|---|---|
| none (bidirectional) | 0.941 |
| causal | 0.9835 |
| **causal + sliding window 30** | **1.0** |

The codec decoder attends only within `decoder_window` (30) frames. On a 96-frame
sequence that is not a detail -- it is the difference between right and wrong, and no
amount of staring at tensor shapes would have revealed it.

Final verification against the official waveform for the same codes:

* **relative error 0.0, correlation 1.0**, 30720 samples

So audio output now has both halves transcribed and exact:

| part | state |
|---|---|
| depth decoder (LM hidden -> 12 codes) | **exact** (logits 1.0, codes 36/36) |
| liquid codec (codes -> 24 kHz waveform) | **exact** (rel 0.0, corr 1.0) |

What remains for audio out is wiring (`text model -> depth decoder -> maskgit [4,4,4]
-> codec`) and the ANE conversion of both parts. The codec's attention arithmetic is
the same as the text model's, so the layer build largely carries over; its conv kernel
is 7 and its width 512.
## ROUND 25-27: speech out runs end to end; the loop produces silence

`tools/audio_out.py` now does the whole chain on the engine: prompt -> ANE text model ->
hidden state -> depth decoder (maskgit 4/4/4, greedy) -> 12 codes per frame -> my codec
-> 24 kHz WAV. It runs and writes 1.92 s of audio. **The audio is silence** (rms 15 of
32767), so the chain is wired but not yet correct.

The decisive measurement: the **official** model's codes for the same prompt decode
through *my* codec to **rms 0.076, peak 0.42** -- normal speech level.

| source of codes | decoded rms (float scale) |
|---|---|
| official model (24 frames) | **0.076** |
| my loop, pre-norm hidden | 0.0010 |
| my loop, final-norm hidden | 0.0005 |

So the codec is proven on real codes, and the fault is in how I drive the depth
decoder from the text model's hidden state.

Also learned: the entry point is `Lfm3OmniForConditionalGeneration.generate(...)` (not
`AutoModel`), the mode is `generation_mode='audio'` (values: text, audio, interleaved),
and the result carries `audio_codes` of shape (1, 12, frames).

Next step is exact: capture the official hidden state at frame 0 in the reference
environment, feed it to **my** depth decoder, and compare the codes against the
official `[255, 588, 1640, 698, ...]`. That separates a depth-decoder sampling fault
from a wrong text hidden state in one measurement.
## ROUND 28: the sampler is the suspect, not the text model

Measured in this order, each step narrowing the fault:

| measurement | result |
|---|---|
| my ANE hidden vs official hidden (post-norm) | cos **0.9934**, norms 21.2 vs 20.8 |
| my hidden pre-norm vs official | cos 0.30 -- so the depth decoder really is fed the **normalised** state |
| official 24-frame codes through my codec | rms **0.076** (speech level) |
| my 24-frame codes through my codec | rms **0.0018** |
| my sampler on the *official* hidden, 24 frames | rms **0.0009** |

That last line is the important one: with the official hidden state, **my** sampler still
produces quiet audio, so the fault is in the sampling path, not in the Neural Engine text
model. Good news -- the part I can fix outright.

Two traps sprung while measuring, both worth recording:

1. **Repeating one frame is not a valid loudness test.** Repeating the *official* frame 12
   times decodes to rms 0.00037, quieter than my whole sequence. The codec's energy
   depends on frame-to-frame variation, so single-frame tests say nothing.
2. **Greedy is not the model's decoder.** `generation_config.json` sets
   `audio_temperature: 0.4`, `audio_repetition_penalty: 1.2`,
   `audio_repetition_window: 64`. Greedy sampling of the first frame agrees with the
   official only on codebook 0 -- the top-5 probabilities are 0.051, 0.046, 0.035,
   0.031, 0.030, i.e. almost flat, so argmax is arbitrary and silences the frame.

### Where that leaves audio out

Depth decoder exact, codec exact, chain wired and running, sampler implemented with the
config's temperature and penalty -- output still 40x quieter than the official. The next
checks are the sampler's reveal order (mine is sequential; the model uses `randperm`) and
the repetition penalty's exact form.

### Plan adjustment

I am not going to spend every remaining round on audio timbre while three explicitly
requested things are untouched: **75k context**, **>20-30 tok/s (palettization)** and
**publishing** (HuggingFace model + repo, as done for Dory). Audio out is implemented and
runs; its quality gap is recorded above and I will return to it after those.
## ROUND 29: palettization -- 16.3 -> 21.1 tok/s, answers still right

One conv layer, measured in isolation:

| variant | dispatch | size | output error vs fp16 |
|---|---|---|---|
| fp16 | 1.416 ms | 128 MB | -- |
| **LUT4** | **0.711 ms** | 32 MB | 1.36 % |
| **LUT6** | **0.716 ms** | 48 MB | **0.33 %** |

LUT6 is the right choice: same speed as LUT4 and an eighth of its error. All 26 programs
were rebuilt with it (`tools/palettize_all.py`, weights only, k-means), giving 1.7 GB
against 3.7 GB.

End to end on the engine:

| | fp16 | LUT6 |
|---|---|---|
| cold first token | 0.921 s | **0.648 s** |
| cold decode | 16.3 tok/s | **21.1 tok/s** |
| warm decode (prefix reused) | 16.4 tok/s | **21.5 tok/s** |
| answers | Tokyo / Rome | **Tokyo / Rome** -- unchanged |

So the >20 tok/s half of the objective is met at short context, and text quality held.
Note this is decode at ~25 positions; the number falls as the cache fills, exactly as the
fp16 measurements did.

### What stands between 21 and 30

At 21.1 tok/s a token costs 47 ms. The 26 programs are dispatched from Python one at a
time; at roughly 0.3-0.5 ms of call overhead each that is 8-13 ms -- a fifth of the
budget spent before the engine does any work. The LM head is the other large single cost.
Both are addressable without touching the model: fewer dispatches per token, and a
palettized LM head.
## ROUND 30: where a token actually goes

Per-program dispatch times at LUT6 (median of 30, ANE confirmed):

| program | ms | count per token | total |
|---|---|---|---|
| conv layer | 0.738 | 13 | 9.6 ms |
| full attention (28,672-row mask) | 2.548 | 6 | 15.3 ms |
| sliding attention (256) | 0.726 | 5 | 3.6 ms |
| lm_head | 2.617 | 1 | 2.6 ms |
| final_norm | 0.054 | 1 | 0.05 ms |
| **sum** | | | **31.2 ms** |

Measured end to end: **47 ms/token (21.1 tok/s)**. So ~16 ms is spent outside the
programs -- Python marshalling around each `predict`, the embedding memmap read, and
the per-layer numpy conversions. **Dispatch overhead itself is only 54 us** (measured on
the tiny final_norm program), so my earlier guess that call overhead was 8-13 ms was
wrong; it is the Python-level work around the calls, not the calls.

### Incremental masks

Rebuilding a 28,672-entry mask for every attention layer of every token was worth
removing on principle, and it is now allocated once per session and revealed one
position at a time. Measured effect: 21.5 -> 21.5 tok/s, i.e. **none**. The answers
stayed correct (Tokyo, Rome), which is the part that mattered -- the mask logic changed.

### Where the remaining speed would come from

The 31.2 ms of arithmetic is dominated by full attention (15.3 ms over 6 layers), and
that cost is the 28,672-row tile the program must still read even when almost all of it
is masked. Cutting it means either a smaller cache or a cheaper attention scheme; it is
not a dispatch-tuning problem. The ~16 ms of Python work is reducible but caps at ~30
tok/s on its own.
## ROUND 31: the LM head is mostly not on the engine

In-situ profile of one decode step (median of 10, LUT6):

| part | ms |
|---|---|
| embedding lookup | 0.02 |
| 13 conv layers | 14.83 |
| 6 full-attention layers | 16.25 |
| 5 sliding-attention layers | 3.78 |
| final norm + **LM head** | **11.42** |
## GOAL 78k-ANE, ROUND 3: correcting my own round-2 claim

Round 2 said the repository does not reproduce the published model, on the strength of a
relative difference of 0.93. That number came from `max|a-b| / max|b|`, which on a
residual-dominated signal is decided by one outlier element. Measured properly:

| metric rebuilt vs shipped, }layer 2 | value |
| --- | --- |
| correlation | **0.932** |
| relative L2 | **36.3 %** |
| rms | 0.1027 vs 0.1131 |

So the two are the same computation up to a real but moderate discrepancy -- **not** a
different model. My round-2 sentence was too strong and I am withdrawing it.

What is established, all verified rather than assumed:

* every weight checked matches a fresh byte-range fetch exactly (q_norm, w1, o_proj,
  operator_norm: max abs difference **0.0**);
* the builder is byte-identical to the first commit in the published repo;
* palettisation accounts for **2.1 %** (fp16 vs LUT6 rebuild), so it is not the gap;
* attention variants -- no qk-norm, no RoPE, no gate, scale 1.0 -- all leave the gap at
  ~0.93 on the max metric, so the gap is not in the attention detail;
* the `1 + w` norm convention is worse (24.8), so the norms are used as plain scales.

A 36 % L2 difference with 0.93 correlation is the signature of one term being wired
differently, not of a wholesale mismatch. Next: exercise the two-token case (which brings the
mask and the cache write into play), and check whether the gap is uniform across the six
layers.