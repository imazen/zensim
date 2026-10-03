import json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
reg=json.load(open('/home/lilith/work/zensim-validation-2026-09-08/max-attribution/PIXEL_REGISTER.json'))['inputs']
arm=sys.argv[1]; blocks=[int(b) for b in sys.argv[2].split(',')]
ms=[f'/var/tmp/steercheck/screen/human-{arm}-h128-full-s{s}.bin' for s in (5101,5103,5107)]
tool='/var/tmp/steercheck/target/release/examples/diffmap_block_coherence'
env=dict(os.environ,RAYON_NUM_THREADS='1',ZENSIM_FORMULA_REV='3')
def one(j):
    c,b=j; out=f'/var/tmp/steercheck/v2diag_after/{arm}-{c["index"]}-b{b}'
    if os.path.exists(out+'.diag.json'): return
    e=dict(env,ZENSIM_V2_DIAG=out+'.diag.json')
    subprocess.run([tool,c['reference'],c['decoded'],'--block',str(b),'--json',out+'.json','--ensemble',','.join(ms),'--ensemble-weights','0.3333333333333333,0.3333333333333333,0.3333333333333333'],env=e,stdout=open(out+'.log','w'),stderr=subprocess.STDOUT,check=True)
with ThreadPoolExecutor(5) as p: list(p.map(one,[(c,b) for c in reg for b in blocks]))
