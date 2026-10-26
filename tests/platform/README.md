# Platform selection integration checks

These checks use the packaged Linux x86_64 CPU runtime.

```sh
bazel test //tests/platform:selection_test --test_output=all
# Also exercise CUDA control and a real CUDA-to-CPU fallback:
bazel test //tests/platform:selection_test --@zml//platforms:cuda=true --test_output=all
```

The probe compiles and evaluates a small integer computation, checking 42 and
the selected backend. No model download is needed. CPU control always runs;
CUDA control requires a CUDA-enabled build and an exposed NVIDIA device.

On Linux, bubblewrap isolates `/dev` and replaces or hides the CPU plugin only
inside child mount namespaces. Missing, corrupt and permission-denied plugins
are checked under automatic and explicit CPU selection. The tests verify that
the program reached selection, the expected loader cause and propagated error,
and absence of a selected backend. Timeouts are test errors.

Isolation cases are explicitly skipped if bubblewrap/user namespaces are
unavailable. The permission-denial group requires an unprivileged user. Read
the test output to distinguish exercised scenarios from skips. These tests do
not simulate a PJRT client that initializes successfully with zero devices.
