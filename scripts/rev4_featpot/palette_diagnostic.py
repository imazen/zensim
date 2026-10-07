"""Label-free PALETTE diagnostics: summarize exact extraction outputs and plot curves.

The CHROMAQ multiplier is a quantization-table knob, not a saturation gain.
This diagnostic makes no prediction-quality or model-adoption claim.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    a = parser.parse_args()
    colour = a.root/'colour_edits.csv'
    chromaq = a.root/'chromaq/curves.csv'
    edits = list(csv.DictReader(colour.open()))
    curves = list(csv.DictReader(chromaq.open()))
    assert len(edits) == 385, 'complete 55-edit, seven-N diagnostic required'
    groups = {}
    for r in curves:
        groups.setdefault((r['ladder'], int(r['N'])), []).append(r)
    directions = []
    for kind, centre, signal in [('chroma',1,'chroma_signed'),('lightness',0,'lightness_signed'),('hue',0,'hue_signed')]:
        rows=[r for r in edits if r['edit']==kind and abs(float(r['level'])-centre)>1e-12]
        good=sum(float(r[signal])*(float(r['level'])-centre)>0 for r in rows)
        directions.append({'edit':kind,'direction_agreement':good,'direction_tested':len(rows),
            'note':'Full diagnostic grid, including large hue rotations; not an accuracy/adoption test.'})
    summary = []
    for (ladder, n), rows in sorted(groups.items()):
        rows.sort(key=lambda r: float(r['multiplier']))
        summary.append({'ladder':ladder, 'N':n, 'cells':len(rows),
            'pairs':sum(int(r['pairs']) for r in rows),
            'multiplier_min':float(rows[0]['multiplier']), 'multiplier_max':float(rows[-1]['multiplier']),
            'mean_shift_median_first':float(rows[0]['mean_shift_median']),
            'mean_shift_median_last':float(rows[-1]['mean_shift_median']),
            'signed_chroma_median_min':min(float(r['chroma_signed_median']) for r in rows),
            'signed_chroma_median_max':max(float(r['chroma_signed_median']) for r in rows)})
    result = {'schema':'palette-diagnostic-v1', 'labels_read':False,
        'colour_edit_rows':len(edits), 'colour_edit_directions':directions, 'chromaq_curve_cells':len(curves),
        'inputs':{str(p):digest(p) for p in (colour,chromaq)},
        'classification':'CHROMAQ owner; multiplier changes quantization, not chroma gain',
        'curves':summary,
        'limitations':['Spatially agnostic distributions; no spatial misalignment map.',
            'RGB8 edits are quantized and may clip at the gamut boundary.',
            'Median-cut repartition and assignment can make curves nonmonotonic and flip hue direction, including small rotations.',
            'No human labels, fits, accuracy claim or serving adoption.']}
    local_hue=[r for r in edits if r['edit']=='hue' and 1e-12<abs(float(r['level']))<=0.3]
    result['hue_direction_within_0_3_radians']={'agreement':sum(float(r['level'])*float(r['hue_signed'])>0 for r in local_hue),'tested':len(local_hue)}
    out=a.root/'summary.json'
    with out.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ladders=sorted({k[0] for k in groups})
    fig,axes=plt.subplots(len(ladders),2,figsize=(12,3*len(ladders)),squeeze=False)
    for row,ladder in enumerate(ladders):
        for n in range(2,9):
            rows=sorted(groups[(ladder,n)],key=lambda r:float(r['multiplier']))
            x=[float(r['multiplier']) for r in rows]
            axes[row,0].plot(x,[float(r['mean_shift_median']) for r in rows],marker='.',label=f'N={n}')
            axes[row,1].plot(x,[float(r['chroma_signed_median']) for r in rows],marker='.',label=f'N={n}')
        for col,signal in enumerate(('weighted colour shift','signed chroma shift')):
            ax=axes[row,col];ax.set_title(f'{ladder}: median {signal}')
            ax.set_xscale('symlog',linthresh=1);ax.set_xlabel('CHROMAQ quantization multiplier')
            ax.axhline(0,color='black',linewidth=.5);ax.grid(alpha=.2)
    axes[0,0].legend(ncol=4);fig.tight_layout()
    fig.savefig(a.root/'chromaq/curves.png',dpi=150)
    fig.savefig(a.root/'chromaq/curves.pdf')
    print(json.dumps({'edit_rows':len(edits),'curve_cells':len(curves),'ladders':ladders}))


if __name__ == '__main__':
    main()
