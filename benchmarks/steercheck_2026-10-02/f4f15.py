import pyarrow.parquet as pq, numpy as np, json, glob, sys
def cols(ids): return [f'f{i}' for i in ids]
frag=[372+s*87+c*29+21 for s in range(4) for c in range(3)]
band=[372+s*87+c*29+27 for s in range(4) for c in range(3)]
masked_iw=list(range(228,372))
out={}
tabs={'rev3_human_fit':'/home/lilith/work/zensim-validation-2026-09-13/steerable/screen/human_fit.parquet',
 'rev3_codec_fit':'/home/lilith/work/zensim-validation-2026-09-13/steerable/screen/codec_fit.parquet',
 'rev3_corruption_fit':'/home/lilith/work/zensim-validation-2026-09-13/steerable/screen/corruption_fit.parquet'}
for p in sorted(glob.glob('/var/tmp/rev4-featpot/v2c/wide/main/real/*_fit.parquet')): tabs['rev4_'+p.split('/')[-1][:-8]]=p
def stats(a):
    a=np.asarray(a,dtype=np.float64); f=np.isfinite(a)
    return dict(n=int(a.size),nonfinite=int((~f).sum()),min=float(a[f].min()),max=float(a[f].max()),p99=float(np.percentile(a[f],99)),p999=float(np.percentile(a[f],99.9)))
for name,p in tabs.items():
    names=set(pq.read_schema(p).names)
    t=pq.read_table(p,columns=[c for c in cols(masked_iw+frag+band) if c in names])
    d={c:t[c].to_numpy(zero_copy_only=False) for c in t.column_names}
    r={'rows':t.num_rows}
    mx=[(i,np.nanmax(np.abs(d[f'f{i}']))) for i in masked_iw if f'f{i}' in d]
    r['f313']=stats(d['f313'])
    r['masked_iw_overall_max_abs']=max(m for _,m in mx)
    r['masked_iw_top5']=sorted(mx,key=lambda x:-x[1])[:5]
    r['masked_iw_cols_over_1e3']=[i for i,m in mx if m>1e3]
    r['masked_iw_cols_over_10']=[i for i,m in mx if m>10]
    r['fragility']={f'f{i}':stats(d[f'f{i}']) for i in frag}
    r['fragility_all_exactly_1']=bool(all((d[f'f{i}']==1.0).all() for i in frag))
    r['fragility_distinct_counts']={f'f{i}':int(len(np.unique(d[f'f{i}']))) for i in frag}
    r['banding_range']=[float(min(np.nanmin(d[f'f{i}']) for i in band)),float(max(np.nanmax(d[f'f{i}']) for i in band))]
    out[name]=r
json.dump(out,open('f4f15.json','w'),indent=1,default=str)
for n,r in out.items(): print(n,r['rows'],'f313',r['f313']['min'],r['f313']['max'],'maskediw max',r['masked_iw_overall_max_abs'],'>1e3',r['masked_iw_cols_over_1e3'][:8],'frag exact1',r['fragility_all_exactly_1'],'frag distinct',list(r['fragility_distinct_counts'].values())[:4])
