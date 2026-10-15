"""Build the zero-update scalar wrapper with the task's one zenpredict identity."""
import json
from pathlib import Path
import subprocess

base=Path(__file__).resolve().parents[2]
metadata=json.loads(subprocess.check_output(['cargo','metadata','--locked','--format-version','1'],cwd=base))
packages=[p for p in metadata['packages'] if p['name']=='zenpredict' and '417cc785b83bee0b3f827ae85c5203f5e104a308' in (p['source'] or '')]
assert len(packages)==1
source=str(Path(packages[0]['manifest_path']).parent)
config='[patch."https://github.com/imazen/zenanalyze"]\nzenpredict = { path = '+json.dumps(source)+' }\n'
path=Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part5/jxl-build-patch.toml')
if path.exists():assert path.read_text()==config
else:
    with path.open('x') as f:f.write(config)
subprocess.run(['cargo','build','--manifest-path',str(base/'tools/e34-jxl-scalar/Cargo.toml'),'--config',str(path)],check=True,cwd=base)
