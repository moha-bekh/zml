# Troubleshooting a Linux CPU deployment

Keep the executable and its complete runfiles together when deploying a Bazel
build. A Linux distribution name alone does not determine whether its dynamic
loader and libraries can load the CPU runtime.

## Identify the missing dependency

For an extracted Linux x86_64 deployment, set these paths to the actual files:

```sh
app="$PWD/mnist"
runtime_lib="$PWD/mnist.runfiles/_main/platforms/cpu/sandbox/lib"
getconf GNU_LIBC_VERSION
readelf -l "$app"
readelf -d "$runtime_lib/libzml_cpu.so"
readelf --version-info "$runtime_lib/libzml_cpu.so"
readelf --version-info "$runtime_lib/libunwind.so.1"
ldd "$runtime_lib/libzml_cpu.so"
```

Inspect the `Version needs` entries and the text from `ldd`, which can return
zero even when it reports missing symbol versions. Check the libraries shipped
beside the plugin as well as the plugin itself. Runfiles paths can differ by
workspace; use the paths from your deployment rather than assuming `_main`.

For example, the Linux amd64 CPU archive pinned in
[`platforms/cpu/cpu.bzl`](../../platforms/cpu/cpu.bzl) at ZML
`b71982fda7ef733e98362520b7cfe62e33eee985` is
`202610231457.291.1.799ad5c39a34` (archive SHA256
`309b737f996024b98b1ee18a44e37df1b328e7c1502a98f5829c10120eb35d63`).
Its ELF requirements include:

| Component | Highest required versions in the inspected bundle |
| --- | --- |
| `libzml_cpu.so` | `GLIBC_2.29`, `GLIBCXX_3.4.30`, `CXXABI_1.3.13` |
| Bundled `libunwind.so.1` | `GLIBC_2.34` |

Thus inspecting only the plugin understates the glibc requirement of this
bundle. These are requirements of the identified artifacts, not a general
minimum for every ZML build or an official distribution support matrix.
Recheck the ELF files when the artifact pin changes. This check also does not
establish CPU instruction-set compatibility.

## Launching a packaged binary on NixOS

An ELF interpreter error can occur before any ZML code runs. Inspect the
`Requesting program interpreter` line from `readelf -l "$app"`; a generic
`/lib64/ld-linux-x86-64.so.2` path may not select a usable loader on NixOS.

For a process-local diagnostic, select the loader and GCC runtime from the
installed Nix store outputs. Replace the example store paths below with those
of your environment; do not select arbitrary incompatible library versions.

```sh
glibc_lib=/nix/store/REPLACE-with-glibc-output/lib
gcc_lib=/nix/store/REPLACE-with-gcc-runtime-output/lib
export RUNFILES_DIR="$PWD/mnist.runfiles"
"$glibc_lib/ld-linux-x86-64.so.2" \
  --library-path "$glibc_lib:$gcc_lib:$runtime_lib" \
  "$app" /path/to/mnist.safetensors /path/to/images.idx3-ubyte
```

Keep the application arguments and its runfiles environment intact. This
launch changes library selection for this process; it does not patch the
binary or replace system libraries. If another GCC runtime directory supplies
`libgcc_s.so.1`, include that directory as well.

Resolving the interpreter error alone does not guarantee that the plugin's ABI
requirements are satisfied. Build-time NixOS dependency problems are a separate
case from launching an already packaged executable.
