# Reproduce the Qwen3.5 CPU layer-cache experiment

This branch contains a standalone benchmark for
[`perf/qwen-linear-cache`](https://github.com/moha-bekh/zml/tree/perf/qwen-linear-cache).
The performance PR remains separate: these scripts do not change its CLI or add
an Omniscient dependency to ZML. No model weights, binaries or large recordings
are committed here.

The question is whether passing only the current GatedDeltaNet layer's state
avoids copies of the entire stack of linear-layer states. Measure the compiled
frontend in release_safe, with the same CPU plugin on both refs.

## Run

On Linux with Python3, coreutils, Git and ZML's build prerequisites:

```bash
git clone https://github.com/moha-bekh/zml.git
cd zml
git switch bench/qwen-linear-cache
python3 benchmarks/qwen-linear-cache/fetch_model.py benchmarks/qwen-linear-cache/model
python3 benchmarks/qwen-linear-cache/compare.py \
  --baseline origin/master --candidate origin/perf/qwen-linear-cache \
  --model benchmarks/qwen-linear-cache/model \
  --out benchmarks/qwen-linear-cache/runs/comparison-001
```

The download is the public Qwen3.5-0.8B checkpoint pinned to
`2fc06364715b967f1860aea9cf38778875588b17` (about1.76GB). `model.json` pins the
files and SHA256 digests. An existing matching model directory can be passed
instead. The setup verifies all model hashes before building or measuring.

The driver uses the repository's Bazel launcher by default. `--bazel /path/bazel`
selects an existing installation. `--output-base /path/cache` reuses an existing
Bazel build directory; both builds are serialized and their binaries/runfiles
are preserved before switching workspace. An initial ZML build requires LLVM,
MLIR, compiler dependencies and substantial build time/disk space.

The output directory must be new. Its detached Git worktrees, binaries and
runfiles are intentionally retained for inspection. They are not Git commits;
`git worktree remove PATH` can remove each scratch worktree once no longer needed.
Use a dedicated output base when another agent or build is using the workspace.

To repeat timings with the exact preserved binaries, without rebuilding:

```bash
python3 benchmarks/qwen-linear-cache/compare.py \
  --reuse-builds benchmarks/qwen-linear-cache/runs/comparison-001 \
  --model benchmarks/qwen-linear-cache/model \
  --out benchmarks/qwen-linear-cache/runs/comparison-002
```

This reuses the recorded revisions, verifies binary/CPU-plugin hashes, and
requires the original artifact directories to remain available. It does not
silently follow newer branch heads.

## What is measured

- CPU backend, one **PJRT device** using multiple OS threads, StableHLO attention.
- The same prompt, seqlen64, topk1 and at most8 generated tokens (EOS may stop it).
- Model compilation and weights loaded once per process; a fresh session/KV
  cache for each generation. Turn0 is excluded as an in-process warmup.
- Three independent process pairs, alternating order. Each process has three
  measured generations after its warmup. Report process medians, not nine
  independent process samples.
- Exact comparison to seven archived token IDs for this pinned model/prompt.
  These short checks do not establish answer quality or long-context correctness.

The common `harness.patch` applies identically to both refs. It adds bounded
repetition, CPU1 selection and phase markers; native console IO avoids a
separate VFS console failure. The production performance change is not part of
this harness. For a different model/revision, deliberately establish another
reference; do not silently disable a failed functional check.

`summary.json` reports medians of process medians and the relative decode
**duration** reduction. Decode includes host synchronization, token output and
the terminal EOS evaluation; this is not a kernel benchmark. Session setup and
prefill are reported separately from decode. The turn timer excludes destruction
of the session, while the recorded process lifetime includes teardown.

Cache state, boost/frequency and external load are uncontrolled. Run on an idle
machine and inspect dispersion and every process result before interpreting a
small difference. A negative result must remain visible; a gain is not promised
on every CPU/model/sequence length. This target uses short single-user generation,
not continuous batching or a multi-request server.

## Optional trace as code with Omniscient

Add `--omn /absolute/path/to/omn` to `compare.py`. After **all** uninstrumented
pairs finish, it records one diagnostic profile per variant through a concrete
native JSON session. Both processes execute the same number of turns.

The stored requests ask for `cpu.profile` at199Hz, process CPU time and RSS.
The versioned `session.json` is the trace-as-code template; `run.py` binds its
launch target and saves the executable session with each recording. Each recording preserves the resolved backend, loss/degradation information,
collector cost and separate program/collector stdout/stderr. Use the Omn version
and binary hash from each manifest; this is a separate, optional tool.

```bash
omn diff \
  benchmarks/qwen-linear-cache/runs/comparison-001/profile-baseline-0/trace \
  benchmarks/qwen-linear-cache/runs/comparison-001/profile-candidate-0/trace
```

Inspect known `CopyThunk::Execute` stacks and the HLO before/after. Anonymous JIT
addresses may change between processes, kernel symbols may be unavailable, and
late ticks/stack truncation must be reported. The automatic differential ranking
of an unknown address is not an identified function. Profiled times include tool
perturbation and must not replace the uninstrumented timing comparison.

To collect HLO, use a **separate diagnostic run** with XLA dump flags and preserve
its environment. Do not turn compilation/dump IO into an inference measurement.

## Scope before upstreaming

The earlier research prototype showed a16.5% decode-duration reduction on a
Ryzen9 3900X, on this short CPU1 workload; other runs varied, including an initial
arithmetic case with no gain. This number is context, not a result guaranteed
by this branch. Publish the new run's exact refs, environment and raw process
medians when reviewing the cleaned-up implementation.

The performance branch also contains numerical no-checkpoint tests comparing
monolithic and per-layer states through real GatedDeltaNet forwards, including
untouched layers. Long conversations, larger models, multi-GPU and other
accelerators still require validation before treating this as generally ready.
