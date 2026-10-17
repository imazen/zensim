"""Retain historical publications plus the explicitly amended whole phase."""
import json
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
