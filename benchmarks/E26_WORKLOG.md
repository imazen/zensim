# E26 worklog

Part A landed as-is at main ef31614a77db7788d8aea34715d8f1030e5e0ab9,
as confirmed by the coordinator. Part B is its direct child; no rebase.

Before any fit: fixed q_jod transform and explicit native research caller
registered in the original E26 record. No E26 fit has started. Full decision
rule, arms, feature IDs, seeds, folds, and VAL population remain unchanged.

The existing shared fleet queue is not host scoped: host_filler.sh invokes
kids_pick.py with QUEUE=fleet_queue; kids_pick.py reads unqualified triples.
It serves prohibited hosts as well as allowed hosts. E26 will not enter this
queue until its allowed-host restriction is enforceable. Existing workers
are owned by other lanes and are not stopped or restarted.

Coordinator resolved placement: only the five allowed consumers run, all
single-thread cells; existing tower capacity explicitly accepted. No extra
filler or worker will be started/restarted. Transform and rank-only leg
committed before fitting at 8e8f3170.
