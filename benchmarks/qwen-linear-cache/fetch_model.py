#!/usr/bin/env python3
"""Fetch the pinned public checkpoint, checking every file before installation."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directory',type=Path)
a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
manifest=json.loads(Path(__file__).with_name('model.json').read_text())
for row in manifest['files']:
    destination=a.directory/row['file']
    if destination.exists():
        with destination.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=row['sha256']:raise SystemExit(f'Existing file has wrong checksum, not overwritten: {destination}')
        print('verified',destination,flush=True)
        continue
    digest=hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=a.directory,prefix='.download-',delete=False) as output:
        temporary=Path(output.name)
        try:
            with urllib.request.urlopen(row['url'],timeout=120) as response:
                while chunk:=response.read(1024*1024):
                    digest.update(chunk);output.write(chunk)
            output.flush()
            if digest.hexdigest()!=row['sha256']:raise RuntimeError(f'Checksum mismatch: {row["file"]}')
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(destination)
    print('downloaded and verified',destination,flush=True)
