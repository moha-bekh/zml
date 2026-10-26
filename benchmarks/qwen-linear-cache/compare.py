#!/usr/bin/env python3
"""Build two refs with a common bounded harness, then alternate unprofiled CPU runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def copy_reflink(source, destination):
    subprocess.run(['cp', '--reflink=auto', '--preserve=mode,timestamps', str(source), str(destination)], check=True)
    return str(destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default='origin/master')
    p.add_argument('--candidate', default='origin/perf/qwen-linear-cache')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--reuse-builds', type=Path, help='Reuse and verify artifacts from a completed comparison instead of rebuilding')
    p.add_argument('--out', type=Path, required=True, help='New output directory, including preserved build artifacts')
    p.add_argument('--bazel', type=Path, default=ROOT / 'bazel.sh')
    p.add_argument('--output-base', type=Path, help='Optional existing Bazel output base; builds are serialized')
    p.add_argument('--output-user-root', type=Path, help='Optional Bazel cache root matching an existing output base')
    p.add_argument('--pairs', type=int, default=3)
    p.add_argument('--turns', type=int, default=4)
    p.add_argument('--jobs', type=int, default=6)
    p.add_argument('--omn', type=Path, help='Optional separate CPU profiles after all uninstrumented pairs')
    a = p.parse_args()
    if a.pairs < 1 or a.turns < 2:
        p.error('Need at least one pair and two turns (first turn is warmup)')
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    model = a.model.resolve()
    expected = json.loads((HERE / 'model.json').read_text())
    for row in expected['files']:
        f = model / row['file']
        if not f.is_file() or sha(f) != row['sha256']:
            p.error(f'Model file missing or checksum mismatch: {f}')
    meta = dict(baseline_ref=a.baseline, candidate_ref=a.candidate, model=expected,
                uname=platform.uname()._asdict(), affinity=sorted(os.sched_getaffinity(0)),
                pairs=a.pairs, turns=a.turns, builds={}, runs=[],
                caveat='Shared host; page cache, frequency and other workloads uncontrolled. CPU1 is a PJRT device, not one OS thread.')
    write(out / 'experiment.json', meta)
    artifacts = {}
    output_base = a.output_base.resolve() if a.output_base else out / 'bazel-output'
    if a.reuse_builds:
        previous = a.reuse_builds.resolve()
        prior = json.loads((previous / 'experiment.json').read_text())
        meta['builds'] = prior['builds']
        meta['baseline_ref'] = prior['baseline_ref']
        meta['candidate_ref'] = prior['candidate_ref']
        meta['reused_builds_from'] = str(previous)
        for label in ['baseline', 'candidate']:
            folder = previous / ('artifact-' + label)
            binary, runfiles = folder / 'llm', folder / 'llm.runfiles'
            build = meta['builds'][label]
            if sha(binary) != build['binary_sha256']:
                raise RuntimeError('Preserved executable checksum mismatch: ' + label)
            for name, digest in build['backend_files'].items():
                if sha(runfiles / name) != digest:
                    raise RuntimeError('Preserved CPU plugin checksum mismatch: ' + name)
            artifacts[label] = (binary, runfiles, build['revision'])
        write(out / 'experiment.json', meta)
    else:
        for label, ref in [('baseline', a.baseline), ('candidate', a.candidate)]:
            revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', ref + '^{commit}'], text=True).strip()
            source = out / ('source-' + label)
            subprocess.run(['git', '-C', str(ROOT), 'worktree', 'add', '--detach', str(source), revision], check=True)
            subprocess.run(['git', '-C', str(source), 'apply', '--check', str(HERE / 'harness.patch')], check=True)
            subprocess.run(['git', '-C', str(source), 'apply', str(HERE / 'harness.patch')], check=True)
            patch = subprocess.check_output(['git', '-C', str(source), 'diff'], text=True)
            (out / (label + '-harness.diff')).write_text(patch)
            command = [str(a.bazel.resolve()), '--output_base=' + str(output_base), 'build', '//examples/llm',
                       '--@zml//platforms:cuda=false', '--@zml//platforms:cpu=true',
                       '--@rules_zig//zig/settings:mode=release_safe', '--jobs=' + str(a.jobs)]
            if a.output_user_root:
                command.insert(1, '--output_user_root=' + str(a.output_user_root.resolve()))
            meta['builds'][label] = dict(revision=revision, command=command, cwd=str(source))
            write(out / 'experiment.json', meta)
            with open(out / (label + '-build.log'), 'w') as log:
                subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
            # Preserve artifacts before switching the shared Bazel workspace.
            folder = out / ('artifact-' + label)
            folder.mkdir()
            built = source / 'bazel-bin/examples/llm/llm'
            binary = folder / 'llm'
            copy_reflink(built, binary)
            runfiles = folder / 'llm.runfiles'
            shutil.copytree(str(built) + '.runfiles', runfiles, symlinks=False, copy_function=copy_reflink)
            artifacts[label] = (binary, runfiles, revision)
            meta['builds'][label]['binary_sha256'] = sha(binary)
            meta['builds'][label]['backend_files'] = {str(f.relative_to(runfiles)): sha(f) for f in runfiles.rglob('libzml_cpu.so')}
            write(out / 'experiment.json', meta)
    backends = [sorted(meta['builds'][label]['backend_files'].values()) for label in ['baseline', 'candidate']]
    if not backends[0] or backends[0] != backends[1]:
        raise RuntimeError('CPU plugin artifacts differ; choose refs with the same backend before comparing the frontend change')
    def run(label, index, profile=False):
        binary, runfiles, revision = artifacts[label]
        destination = out / (('profile-' if profile else 'run-') + label + '-' + str(index))
        command = [sys.executable, str(HERE / 'run.py'), '--binary', str(binary), '--runfiles', str(runfiles),
                   '--model', str(model), '--out', str(destination), '--repeats', str(a.turns), '--source-revision', revision]
        if profile:
            command += ['--profile', '--omn', str(a.omn.resolve())]
        subprocess.run(command, cwd=ROOT, check=True)
        result = json.loads((destination / 'result.json').read_text())
        if not result['control_passed']:
            raise RuntimeError('Functional check failed; no performance conclusion')
        meta['runs'].append(dict(variant=label, pair=index, profile=profile, result=str(destination),
                                 metrics=result['median_after_warmup']))
        write(out / 'experiment.json', meta)
    # All builds finish before any timed runs. Each process contains one warmup.
    for pair in range(a.pairs):
        for label in (['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']):
            run(label, pair)
    summary = {label: {key: statistics.median(row['metrics'][key] for row in meta['runs'] if row['variant'] == label and not row['profile'])
                       for key in ['elapsed_seconds', 'prefill_seconds', 'decode_seconds']} for label in artifacts}
    summary['decode_duration_reduction_percent'] = 100 * (1 - summary['candidate']['decode_seconds'] / summary['baseline']['decode_seconds'])
    summary['scope'] = 'Median of independent process medians; within-process turn0 excluded. Whole generation, not GPU kernel timing.'
    write(out / 'summary.json', summary)
    if a.omn:
        for label in artifacts:
            run(label, 0, profile=True)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
