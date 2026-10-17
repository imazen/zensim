"""Install cooperative archive drain methods without replacing active transfers."""
import atexit
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def mirror(root,folder,receipt):
    root=Path(root);folder=Path(folder)
    os.environ['E36_ARCHIVE_ROOT']=str(root)
    safe=module('e36_cooperative_archive',folder/'e36_archive_store.py')
    import e36_archive_store
    for name in ['disconnect','close','hold']:
        setattr(e36_archive_store.ArchiveClient,name,getattr(safe.ArchiveClient,name))
    # Active object transfers keep their handles and data. The next receipt
    # backup reuses one connection and respects previous verified locations.
    updated=module('e36_metadata_archive',folder/'e36_artifact_mirror.py')
    old=sys.modules['__main__']
    if Path(old.__file__).name!='e36_artifact_mirror.py':raise ValueError('exact live archival owner required')
    old.mirror_view=updated.mirror_view
    def finish():
        if updated.METADATA_CLIENT is not None:updated.METADATA_CLIENT.close()
    atexit.register(finish)
    with Path(receipt).open('x') as stream:json.dump(dict(pid=os.getpid(),role='mirror',
        drain_owner=str(folder/'e36_archive_store.py'),metadata_owner=str(folder/'e36_artifact_mirror.py'),
        existing_transfer_handles_retained=True,manual_kills=0,utc=time.time(),fits=0),stream,indent=2)


def executor(root,folder,receipt):
    root=Path(root);folder=Path(folder)
    budget=module('e36_cooperative_executor',folder/'e36_budget4.py');budget.policy()
    import e36_archive_store
    if Path(e36_archive_store.ArchiveClient.disconnect.__code__.co_filename)!=folder/'e36_archive_store.py':raise ValueError('safe future archive client missing')
    with Path(receipt).open('x') as stream:json.dump(dict(pid=os.getpid(),role='executor',
        drain_owner=str(folder/'e36_archive_store.py'),deadline=float(os.environ['E36_GENERATION_DEADLINE']),
        manual_kills=0,utc=time.time(),fits=0),stream,indent=2)
