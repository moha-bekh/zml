# Portable attention on SM75

Run explicitly on a Linux NVIDIA SM75 device (for example RTX 2080 Ti):

```sh
bazel test //tests/attention:portable_sm75_test \
  --@zml//platforms:cuda=true --test_output=all
```

This manual hardware test requires CUDA and rejects other compute capabilities.
It asserts automatic StableHLO selection and checks both the automatic and
forced portable paths against a host f64 softmax(QKᵀ/√64)V calculation.
Positions 0, 31 and 63 cover a single visible key, a partial causal prefix and
the full 64-key sequence. Inputs are deterministic f32 tensors.

There are no model downloads, custom Zig test runner or timing measurements.
This test does not exercise CuTe or certify attention on other GPU families.
