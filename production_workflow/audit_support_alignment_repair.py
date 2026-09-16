"""Independently audit repaired labels and unchanged physics; saved data only."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    a = p.parse_args()
    root = a.workspace.resolve(); workflow = root / 'production_workflow'
    old = workflow / 'gate4_forward_evaluation_20260829_a'
    new = workflow / 'gate4_forward_evaluation_support_aligned_20260910'
    out = workflow / 'support_alignment_correction_20260910'
    out.mkdir(exist_ok=True)
    old_manifest = read(old/'evaluation_manifest.json')
    new_manifest = read(new/'evaluation_manifest.json')
    for path, manifest in [(old,old_manifest),(new,new_manifest)]:
        for name, expected in manifest['output_sha256'].items():
            assert sha(path/name) == expected, (path,name,'hash')
    assert new_manifest['supersedes_manifest_id'] == old_manifest['manifest_id']
    before = pd.read_csv(old/'control_population_metrics.csv')
    after = pd.read_csv(new/'control_population_metrics.csv')
    keys = ['control_id','population','support_stratum']
    assert not before.duplicated(keys).any() and not after.duplicated(keys).any()
    b = before.set_index(keys).sort_index(); n = after.set_index(keys).sort_index()
    whole_b = b.xs('all',level='support_stratum').sort_index()
    whole_n = n.xs('all',level='support_stratum').sort_index()
    pd.testing.assert_frame_equal(whole_b,whole_n,check_exact=False,rtol=1e-12,atol=1e-10)
    exact_whole = whole_b.equals(whole_n)
    numerical = whole_b.select_dtypes(include='number').columns
    max_change = float((whole_b[numerical]-whole_n[numerical]).abs().max().max())
    delta = b[['rows','vector_rmse_m_per_a']].join(n[['rows','vector_rmse_m_per_a']],
                how='outer',lsuffix='_before',rsuffix='_after')
    delta['rmse_change'] = delta.vector_rmse_m_per_a_after-delta.vector_rmse_m_per_a_before
    delta.to_csv(out/'before_after_support_metrics.csv')
    counts=[]
    for (control,population), group in after.groupby(['control_id','population']):
        whole = group.loc[group.support_stratum.eq('all')].iloc[0]
        strata = group.loc[~group.support_stratum.eq('all')]
        assert set(strata.support_stratum).issubset({'neither','marginal_only','joint_only','both'})
        assert strata.rows.sum() == whole.rows
        original = before.loc[before.control_id.eq(control)&before.population.eq(population)]
        counts.append(dict(control_id=control,population=population,rows=int(whole.rows),
                           assigned_before=int(original.loc[~original.support_stratum.eq('all'),'rows'].sum()),
                           assigned_after=int(strata.rows.sum())))
    pd.DataFrame(counts).to_csv(out/'support_partition_checks.csv',index=False)
    gate2 = workflow/'gate2_results/gate2_distribution_diagnostics_20260820_c'
    support_summary = pd.read_csv(gate2/'support_categories.csv')
    median = after.loc[after.control_kind.eq('median')]
    for (experiment,config,population), group in median.groupby(['experiment','configuration','population']):
        expected = support_summary.loc[support_summary.experiment.eq(experiment)
                       & support_summary.configuration.str.startswith(config+'_')
                       & support_summary.population.eq(population)]
        assert len(expected)==1
        total=int(group.loc[group.support_stratum.eq('all'),'rows'].iloc[0])
        assert total==int(expected.iloc[0].heldout_rows)
        for cat in ['neither','marginal_only','joint_only','both']:
            fraction=group.loc[group.support_stratum.eq(cat),'rows'].sum()/total
            assert abs(fraction-float(expected.iloc[0][cat+'_fraction']))<1e-12
    dataset = workflow/'gate2_results/gate2_canonical_dataset_20260820_c/canonical_master_dataset.csv.gz'
    ids=pd.read_csv(dataset,usecols=['row_id','common_eligible'])
    ids=ids.loc[ids.common_eligible.astype(bool),'row_id'].astype(str).to_numpy()
    assert len(np.unique(ids))==len(ids)
    category_names={0:'neither',1:'marginal only',2:'joint only',3:'both'}
    maps={}; sentinel=0; changed=0; comparisons=[]
    with np.load(gate2/'point_support_categories.npz') as labels:
        for path in sorted((new/'median_map_data').glob('*.npz')):
            ensemble=path.stem.removesuffix('_MEDIAN'); experiment,config=ensemble.rsplit('_',1)
            with np.load(path) as corrected,np.load(old/'median_map_data'/path.name) as previous:
                assert corrected.files==previous.files
                for field in corrected.files:
                    if field!='support_category':
                        np.testing.assert_array_equal(corrected[field],previous[field],err_msg=f'{ensemble}/{field}')
                row_ids=corrected['row_id'].astype(str)
                assert len(np.unique(row_ids))==len(row_ids)
                source_ids=ids[labels[experiment+'__row_index']]
                label_key=next(k for k in labels.files if k.startswith(experiment+'__'+config+'_'))
                lookup=pd.Series(labels[label_key],index=source_ids)
                expected=lookup.reindex(row_ids)
                assert expected.notna().all()
                np.testing.assert_array_equal(corrected['support_category'],expected.to_numpy(np.uint8))
                sentinel+=int((previous['support_category']==255).sum())
                changed+=int((previous['support_category']!=corrected['support_category']).sum())
                maps[ensemble]={name:corrected[name].copy() for name in corrected.files}
    # Independently corrected later analyses must agree exactly, not be overwritten.
    representation=workflow/'training_representation_diagnostic_20260909_a'
    point=pd.read_csv(representation/'point_diagnostics.csv.gz',usecols=['experiment','configuration','row_id','support_category'])
    for (experiment,config),group in point.groupby(['experiment','configuration']):
        data=maps[experiment+'_'+config]
        lookup=pd.Series(data['support_category'],index=data['row_id'].astype(str))
        expected=lookup.reindex(group.row_id.astype(str))
        assert expected.notna().all()
        np.testing.assert_array_equal(expected.to_numpy(),group.support_category.to_numpy())
        comparisons.append(dict(experiment=experiment,configuration=config,rows=len(group),labels_identical=True))
    pd.DataFrame(comparisons).to_csv(out/'later_diagnostic_alignment_checks.csv',index=False)
    pig=maps['REG_PIG_CFG02']
    local=pig['signed_local_squared_error_improvement']
    pig_check={'area_fraction_lower_error':float(np.mean(local>0)),
               'both_supported_fraction':float(np.mean(pig['support_category']==3)),
               'supported_fraction_net_squared_error_reduction':float(local[pig['support_category']==3].sum()/local.sum())}
    assert round(100*pig_check['area_fraction_lower_error'],1)==78.1
    assert round(100*pig_check['both_supported_fraction'],1)==91.9
    assert round(100*pig_check['supported_fraction_net_squared_error_reduction'],1)==85.6
    # New support-only atlas replaces affected support panels, not the unaffected velocity plots.
    sys.path.insert(0,str(root/'.python_packages'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.patches import Patch
    palette=['#999999','#e69f00','#56b4e9','#009e73']
    atlas=out/'corrected_support_maps';atlas.mkdir(exist_ok=True)
    for experiment in sorted(set(k.rsplit('_',1)[0] for k in maps)):
        items=[(k,v) for k,v in sorted(maps.items()) if k.rsplit('_',1)[0]==experiment]
        fig,axes=plt.subplots(1,len(items),figsize=(3.1*len(items),4.1),squeeze=False)
        for ax,(name,data) in zip(axes.ravel(),items):
            x=data['x']/1000;y=data['y']/1000
            ax.scatter(x,y,c=data['support_category'],s=.5,rasterized=True,
                       cmap=ListedColormap(palette),norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4),linewidths=0)
            ax.set_aspect('equal');ax.set_title(name.rsplit('_',1)[1]);ax.set_xlabel('Easting (km)')
            ax.tick_params(labelsize=8)
        axes[0,0].set_ylabel('Northing (km)')
        fig.suptitle(experiment+' - primary held-out support categories')
        fig.legend(handles=[Patch(color=palette[i],label=category_names[i]) for i in range(4)],
                   loc='lower center',ncol=4,frameon=False)
        fig.tight_layout(rect=(0,.09,1,.93));fig.savefig(atlas/(experiment+'.png'),dpi=180);plt.close(fig)
    report={'status':'passed','old_evaluation_id':old_manifest['manifest_id'],
            'corrected_evaluation_id':new_manifest['manifest_id'],'controls':726,
            'complete_population_partitions':len(counts),'median_maps':len(maps),
            'whole_population_values_exactly_identical':exact_whole,
            'maximum_whole_population_numeric_change':max_change,
            'old_map_entries_with_unassigned_marker_255':sentinel,
            'map_entries_with_changed_category':changed,
            'old_sentinel_meaning':'unassigned category, not affected-row count',
            'map_fields_other_than_support_exactly_identical':True,
            'gate2_support_summary_agreement':True,'later_diagnostic_cases_identical':len(comparisons),
            'PIG_checks':pig_check,'old_and_new_declared_hashes_verified':True,
            'recomputed_physical_solutions':0,'retrained_models':0,
            'source_sha256':sha(__file__)}
    (out/'numerical_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
