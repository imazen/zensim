"""Read feature-only materialization counts without touching human data."""
import argparse
import json
from pathlib import Path
import sqlite3
import time

def progress(pool):
    path=Path(pool)/'physical-index.sqlite3'
    if not path.exists():return dict(utc=time.time(),stage='mandatory receipt admission',fits=0)
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=1) as c:
        return dict(utc=time.time(),physical_features=c.execute('SELECT count(*) FROM features').fetchone()[0],unique_preferences=c.execute('SELECT count(*) FROM pairs').fetchone()[0],materialization_finished=(Path(pool)/'_MANIFEST.json').exists(),fits=0)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('pool',type=Path);a=p.parse_args();print(json.dumps(progress(a.pool)))
