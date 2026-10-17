"""Exact E36 arm/source/seed product; metadata only."""
ARMS=('tv0-c','tv1-c','tv10-c')
SOURCES=('kadid','tid2013','konfig','cid22_a25')


def admit_inventory(cells):
    wanted={(a,'lodo',s,i) for a in ARMS for s in SOURCES for i in range(10)}
    wanted|={(a,'full',None,i) for a in ARMS for i in range(3)}
    seen=set();names=set()
    for c in cells:
        key=(c['arm'],c['kind'],c['heldout'],c['seed_index'])
        suffix=('without_'+c['heldout']+'_s' if c['kind']=='lodo' else 'full_s')+str(c['seed_index'])
        if (key not in wanted or key in seen or c['name'] in names
                or c['name']!='e36-'+c['arm']+'__N/'+suffix):
            raise ValueError('duplicate/unregistered E36 arm/source/seed/name tuple')
        seen.add(key);names.add(c['name'])
    if seen!=wanted or len(cells)!=129:
        raise ValueError('incomplete exact129-cell E36 Cartesian inventory')
    return names
