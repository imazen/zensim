"""Retain historical publications plus the explicitly amended whole phase."""
import json
import gzip
from pathlib import Path
import re
from e36_store import sha


def native_runs(root):
    names=['fallback-v6','svt-additions-v1'];binding=Path(root)/'source-count-amendment-v1/binding.json'
    if binding.exists():
        doc=json.loads(binding.read_text());name=doc['phases']['additions']['run']
        if not re.fullmatch('[a-z0-9-]+',name):raise ValueError('invalid amended native run namespace')
        manifest=doc['phases']['additions']['manifest']
        if sha(manifest['path'])!=manifest['sha256']:raise ValueError('amended archive native manifest changed')
        if name not in names:names.append(name)
    return names


def native_cas_directories(root):
    return [Path(root)/'runs'/name/'blobs' for name in native_runs(root)]


def published_receipt(root,value):
    path=next((directory/value for directory in native_cas_directories(root) if (directory/value).exists()),None)
    if path is None or sha(path)!=value:raise ValueError('published archive eligibility CAS missing or changed')
    document=json.loads(gzip.decompress(path.read_bytes()))
    if document.get('schema')!='e36-native-ladder-receipt-v1' or document.get('complete') is not True:raise ValueError('archive source ladder incomplete')
    return document
