#!/usr/bin/env python3
"""Portable local harness using Linux wait4; no external time executable required.
RSS/minor/major faults are child's (and reaped descendants') kernel rusage;
with Omn this includes collector, so use Omn's per-task RSS for the workload.
"""
import os,subprocess,sys,time,pathlib
if len(sys.argv)<5 or sys.argv[1]!='--output' or sys.argv[3]!='--':sys.exit('Usage: resource-run.py --output FILE -- COMMAND...')
start=time.monotonic();proc=subprocess.Popen(sys.argv[4:]);_,status,usage=os.wait4(proc.pid,0);code=os.waitstatus_to_exitcode(status);proc.returncode=code
pathlib.Path(sys.argv[2]).write_text(f'Measurement source: Linux wait4, launched child and reaped descendants\nElapsed seconds: {time.monotonic()-start}\nMaximum resident set size (kbytes): {usage.ru_maxrss}\nMinor page faults: {usage.ru_minflt}\nMajor page faults: {usage.ru_majflt}\nUser seconds: {usage.ru_utime}\nSystem seconds: {usage.ru_stime}\nExit code: {code}\n')
sys.exit(code if code>=0 else 128-code)
