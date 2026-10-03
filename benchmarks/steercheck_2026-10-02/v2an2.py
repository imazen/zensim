import json,glob,numpy as np,sys
from scipy.stats import spearmanr
names=['SSIM_MEAN','SSIM_DEV2','SSIM_DEV4','ART','DET','MSE','HF_GAIN','HF_LOSS','HF_MAG_LOSS','SSIM_SOFT_PEAK','ART_SOFT_PEAK','DET_SOFT_PEAK','MASKED_SSIM','MASKED_ART','MASKED_DET','MASKED_MSE','IW_SSIM','IW_ART','IW_DET','IW_MSE','PJND_TRANSDUCER','PJND_FRAGILITY','GMS','TRANS_LOW','TRANS_HIGH','BLOCKINESS','RINGING','BANDING','EDGE_WIDTH']
dirn=sys.argv[1]
cases=[]
for f in sorted(glob.glob(dirn+'/v2basic-*-b32.diag.json')):
    d=json.load(open(f)); cases.append(d)
def sig(i): return (i-372)%29 if i>=372 else -1
fam={'blockiness':[25],'ssim_dev':[1,2],'ssim_mean+soft':[0,9,12,16],'hf':[6,7,8],'edge art/det/ms/pjnd':[3,4,5,10,11,13,14,15,17,18,19,20,23,24],'gms/ring/band/edgew':[22,26,27,28]}
rows=[]
for d in cases:
    ids=np.array(d['ids']);P=np.array(d['predicted_fused']);O=np.array(d['observed']);ds=np.array(d['score_delta'])
    sg=np.array([sig(i) for i in ids])
    base=spearmanr(P.sum(0),ds)[0]
    r={'served':spearmanr(np.array(d['refinement_gain']),ds)[0],'sum':base}
    for n,g in fam.items():
        m=np.isin(sg,g); Q=P.copy(); Q[m]=O[m]; r['oracle_'+n]=spearmanr(Q.sum(0),ds)[0]
        # zero this family's prediction (does it help or hurt)
        Z=P.copy(); Z[m]=0; r['drop_'+n]=spearmanr(Z.sum(0),ds)[0]
    m=sg>=0; Q=P.copy(); Q[m]=O[m]; r['oracle_all_v2']=spearmanr(Q.sum(0),ds)[0]
    Q=P.copy(); Q[:]=O; r['oracle_everything(=M2)']=spearmanr(Q.sum(0),ds)[0]
    rows.append(r)
keys=list(rows[0])
print('n',len(rows)); 
for k in keys: print('%-32s median %.3f mean %.3f  n>=0.7: %d'%(k,np.median([r[k] for r in rows]),np.mean([r[k] for r in rows]),sum(r[k]>=.7 for r in rows)))
