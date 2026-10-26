#!/usr/bin/env python3
"""Persistent Qwen CPU experiments; compile/weights once, fresh KV per repeat."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
GOLDEN=[248068,271,248069,271,14556,248046,198]

def sha(path):
    with open(path,'rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--runfiles',type=Path,required=True)
    p.add_argument('--source-revision')
    p.add_argument('--omn',type=Path)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--prompt',default='Reply with one word: hello.')
    oracle=p.add_mutually_exclusive_group()
    oracle.add_argument('--reference',type=Path,help='Baseline result.json for this prompt')
    oracle.add_argument('--record-reference',action='store_true',help='Check repeat consistency only; not an independent correctness oracle')
    p.add_argument('--affinity',help='Comma-separated logical CPU IDs, fixed for workload and collector')
    a=p.parse_args()
    if a.affinity:os.sched_setaffinity(0,{int(x) for x in a.affinity.split(',')})
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=a.binary.resolve();model=a.model.resolve()
    omn=a.omn.resolve() if a.omn else None
    if a.profile and (omn is None or not omn.is_file()):p.error('--profile requires --omn /path/to/omn')
    command=[str(binary),'--model='+str(model),'--prompt='+a.prompt,'--seqlen=64','--topk=1','--backend=stablehlo','--perf-repeats='+str(a.repeats)]
    env=os.environ.copy();env['RUNFILES_DIR']=str(a.runfiles.resolve())
    meta=dict(command=command,cwd=str(ROOT),affinity=sorted(os.sched_getaffinity(0)),
              binary_sha256=sha(binary),omn_sha256=sha(omn) if omn else None,repeats=a.repeats,
              zml_revision=a.source_revision or subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
              model_revision='2fc06364715b967f1860aea9cf38778875588b17',
              load_before=list(os.getloadavg()),cache='uncontrolled page cache; each turn new KV/session; compiled model and weights shared',
              environment={k:env[k] for k in ['RUNFILES_DIR','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','XLA_FLAGS'] if k in env})
    if a.profile:
        session=json.loads(Path(__file__).with_name('session.json').read_text())
        for request in session['requests']:
            request['target']={'launch':command}
        session_path=out/'session.json';session_path.write_text(json.dumps(session,indent=2)+'\n')
        launch=[str(omn),'trace','-f',str(session_path),'--program-stdout',str(out/'program.stdout'),'--program-stderr',str(out/'program.stderr'),'-o',str(out/'trace')]
    else:
        launch=[sys.executable,str(Path(__file__).resolve().with_name('resource_run.py')),'--output',str(out/'resource.txt'),'--',*command]
    meta['launch']=launch;(out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    start=time.monotonic()
    with open(out/('collector.stdout' if a.profile else 'program.stdout'),'w') as stdout,open(out/('collector.stderr' if a.profile else 'program.stderr'),'w') as stderr:
        proc=subprocess.Popen(launch,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        try:code=proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=124
    text=(out/'program.stderr').read_text()+(out/'program.stdout').read_text()
    turns=[]
    expected=json.loads(a.reference.read_text())['turns'][0]['tokens'] if a.reference else GOLDEN
    # Markers bracket initialization + generation; repeat0 is a within-process warmup.
    for match in re.finditer(r'PERF turn_start=(\d+)\n(.*?)PERF turn_end=\1 elapsed_ns=(\d+)',text,re.S):
        index=int(match[1]);body=match[2];tokens=[int(x) for x in re.findall(r'CAMPAIGN generated_token=\d+ token_id=(\d+)',body)]
        prefill=re.search(r'PERF prefill_ns=(\d+)',body)
        decode=re.search(r'decode\x1b\[0m \x1b\[2m([\d.]+)(ms|s)',body)
        turns.append(dict(index=index,elapsed_seconds=int(match[3])/1e9,prefill_seconds=int(prefill[1])/1e9 if prefill else None,
                          decode_seconds=float(decode[1])*({'ms':.001,'s':1}[decode[2]]) if decode else None,
                          tokens=tokens,exact_golden=tokens==expected))
    decode_values=[float(value)*({'ms':.001,'s':1}[unit]) for value,unit in re.findall(r'decode\x1b\[0m \x1b\[2m([\d.]+)(ms|s)',text)]
    if len(decode_values)==len(turns):
        for turn,value in zip(turns,decode_values):turn['decode_seconds']=value
    if a.record_reference and turns:
        expected=turns[0]['tokens']
        for turn in turns:turn['exact_golden']=turn['tokens']==expected
    meta['reference_tokens']=expected
    meta['oracle_kind']='within-process consistency only' if a.record_reference else ('baseline token comparison' if a.reference else 'archived seven-token control')
    (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    valid=code==0 and len(turns)==a.repeats and bool(expected) and all(t['exact_golden'] for t in turns)
    result=dict(exit_code=code,wrapper_seconds=time.monotonic()-start,load_after=list(os.getloadavg()),turns=turns,control_passed=valid,
                instrumentation=a.profile,summary_scope='turn0 warmup excluded; decode includes streaming/host synchronization and terminal EOS evaluation')
    measured=turns[1:]
    if measured:result['median_after_warmup']={k:statistics.median(t[k] for t in measured if t[k] is not None) for k in ['elapsed_seconds','prefill_seconds','decode_seconds'] if any(t[k] is not None for t in measured)}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'sha256.json').write_text(json.dumps({str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()},indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='turns'},indent=2))
    if not valid:raise SystemExit('Workload/functional control failed; no performance conclusion')

if __name__=='__main__':main()
