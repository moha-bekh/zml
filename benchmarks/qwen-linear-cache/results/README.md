# Recorded CPU observations

[ryzen3900x.json](ryzen3900x.json) includes every turn and process median from two
series, model/source/binary/plugin identities, host load and output token checks.
All timings were recorded without Omn. The second series reused the exact same
preserved binaries and runfiles, without rebuilding.

| Series | Baseline decode | Per-layer decode | Duration reduction |
| --- | ---: | ---: | ---: |
| Initial shared-host series | 4.358s | 3.612s | 17.1% |
| Repeat using preserved artifacts | 3.245s | 2.721s | 16.1% |

Each cell is the median of three process medians. Each process has one warmup
and three measured turns, with identical seven-token output. The absolute times
vary substantially with host conditions. The first series includes a pair with
only about1% reduction; it has not been discarded. No universal speedup, GPU
speedup or confidence interval is claimed.

Measured baseline: `b71982fda7ef733e98362520b7cfe62e33eee985`.
Measured candidate: `35288b3d97e2dc7f93e0c5a2dbe5a2e7ce4d12aa`.
Use these commit IDs with `compare.py` to pin the experiment rather than following
future branch heads. Both use the same CPU plugin digest, included in the JSON.

Separate numerical regression tests run real GatedDeltaNet forwards and compare
hidden outputs plus all convolution/recurrent states, including untouched layers:
CPU1/2/4 partitions and CUDA on an RTX2080Ti (SM75). Full and padded prefill,
followed by32 decode steps, use f32 synthetic fixtures with absolute tolerance
1e-6 and relative tolerance1e-5. These are not full-model GPU performance tests.

The pinned full-model CPU benchmark uses BF16 checkpoint weights and compares
token IDs. It does not prove bitwise equality of every hidden state or logit in
a pretrained model. Larger models, long conversations, multi-GPU and other
accelerators remain outside the measured scope.

`hlo-copies.txt` contains the selected recurrent-state copy instructions from
separate diagnostic runs of these same revisions, with digests of the full HLO.
The two18MiB copies become one1MiB copy per linear-layer invocation on this CPU
backend. These sizes count tensor contents in explicit instructions, not total
physical memory traffic. Other tensors, conversions and kernel-internal copies
are outside that count.

`profile-observations.json` reports the known CopyThunk stack group from two
matched diagnostic profiles, both running two generations at199Hz. Each sample
is counted at most once in the inclusive group. Losses and late ticks remain
visible. These profiles validate the packaged session/driver path and support
localization of the copy cost; their timings are excluded from the benchmark.
