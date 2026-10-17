"""Snapshot the owned live archival queue; operational metadata only."""
import json
import os
from pathlib import Path
import sys
import time


def snapshot(root,out):
    root=Path(root);matches={}
    for top in sys._current_frames().values():
        frame=top
        while frame:
            local=frame.f_locals
            if frame.f_code.co_name=='run' and frame.f_globals.get('ROOT')==root and 'dispatcher' in local and 'queue' in local:matches[id(frame)]=frame
            frame=frame.f_back
    if len(matches)!=1:raise ValueError('unique owned archival queue required')
    frame=next(iter(matches.values()));dispatcher=frame.f_locals['dispatcher'];queue=frame.f_locals['queue'];priority=frame.f_locals['priority']
    historical={};unassigned=0;packed=0
    for h in queue:
        assignment=dispatcher.assignments.rows.get(h)
        if assignment is None:unassigned+=1
        elif assignment.get('pack_id'):packed+=1
        else:
            key=assignment['canonical_root'];historical[key]=historical.get(key,0)+1
    result=dict(utc=time.time(),pid=os.getpid(),queued_native_bundles=len(queue),priority_entries=len(priority),
        queued_priority_raw_bytes=sum(-n for n,_ in priority),unassigned_native_bundles=unassigned,pack_assigned_native_bundles=packed,
        historical_assigned_native_bundles_by_root=historical,pending_pack_streams=len(dispatcher.pending),
        cache_reserved_bytes=dispatcher.used,cache_first_active=dispatcher.cache_active,fits=0,qualifies_data=False)
    with Path(out).open('x') as stream:json.dump(result,stream,indent=2)
