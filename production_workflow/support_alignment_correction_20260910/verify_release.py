"""Restricted-diff, unchanged-result, and clean-upload verification."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import csv
import difflib
import hashlib
import json
import re
import subprocess
import pandas as pd


ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'JOG_CANONICAL_SUBMISSION_20260907'
SOURCE=BASE/'manuscript'
BUILD=BASE/'build_support_corrected_20260910'


def sha(p):
    return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()


def main():
    old=(HERE/'source_before/manuscript.tex').read_text(encoding='utf-8')
    new=(SOURCE/'manuscript.tex').read_text(encoding='utf-8')
    checks={}
    equation=lambda s:re.findall(r'\\begin\{equation\}.*?\\end\{equation\}',s,re.S)
    checks['equations_unchanged']=equation(old)==equation(new)
    tables=lambda s:re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',s,re.S)
    checks['tables_unchanged_except_alignment_name']=tables(old.replace('bed--surface alignment','bed--surface gradient alignment'))==tables(new)
    headings=lambda s:re.findall(r'\\(?:sub)*section\{[^}]+\}',s)
    normalized=old.replace('More predictors do not guarantee better transfer','Comparing predictor configurations').replace('Predictor support does not guarantee good forward-velocity performance','Predictor support and forward-velocity performance')
    checks['section_order_preserved']=headings(normalized)==headings(new)
    labels=re.findall(r'\\label\{([^}]+)\}',new)
    refs=re.findall(r'\\(?:ref|nameref|eqref)\{([^}]+)\}',new)
    checks['references_resolve']=not(set(refs)-set(labels)) and len(labels)==len(set(labels))
    cites=lambda s:set(k.strip() for g in re.findall(r'\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}',s) for k in g.split(','))
    bib=(SOURCE/'bibliography.bib').read_text(encoding='utf-8')
    keys=re.findall(r'@\w+\s*\{([^,]+),',bib)
    checks['same_cited_sources']=cites(old)==cites(new)==set(keys) and len(keys)==len(set(keys))
    checks['single_active_tex']=not re.search(r'^\s*\\(?:input|include)\{',new,re.M)
    with (HERE/'figure_hashes_before.csv').open(encoding='utf-8-sig',newline='') as f:
        hashes=list(csv.DictReader(f))
    checks['all_figure_bytes_unchanged']=all(sha(r['Path']).lower()==r['Hash'].lower() for r in hashes)
    with ZipFile(BASE/'Testing_point_wise_inference_of_basal_friction_targeted_corrections_20260910.zip') as z:
        checks['bibliography_class_style_unchanged']=all(z.read(n)==(SOURCE/n).read_bytes() for n in ['bibliography.bib','igs.cls','igs.bst'])
    figure_names=re.findall(r'\\includegraphics(?:\[[^]]*\])?\s*\{([^}]+)\}',new)
    available={p.name for p in (SOURCE/'figures').rglob('*') if p.is_file()}
    checks['all_figure_paths_internal']=all(Path(name).name in available for name in figure_names)
    checks['no_unused_figures']=set(Path(name).name for name in figure_names)==available
    log=(BUILD/'manuscript.log').read_text(encoding='utf-8',errors='replace')
    checks['no_undefined_or_overfull']=not re.search(r'undefined|Overfull|multiply defined',log,re.I)
    (HERE/'manuscript_restricted.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='author_approved_before.tex',tofile='manuscript.tex')),encoding='utf-8')
    # Confirm all aggregate member summaries, not just median controls, are identical.
    old_eval=ROOT/'production_workflow/gate4_forward_evaluation_20260829_a'
    new_eval=ROOT/'production_workflow/gate4_forward_evaluation_support_aligned_20260910'
    member_before=pd.read_csv(old_eval/'ensemble_member_summary.csv')
    member_after=pd.read_csv(new_eval/'ensemble_member_summary.csv')
    index=['ensemble_id','population','support_stratum']
    b=member_before.loc[member_before.support_stratum.eq('all')].set_index(index).sort_index()
    n=member_after.loc[member_after.support_stratum.eq('all')].set_index(index).sort_index()
    pd.testing.assert_frame_equal(b,n,check_exact=True)
    checks['whole_population_member_summaries_exact']=True
    numeric=json.loads((HERE/'numerical_verification.json').read_text())
    checks['support_repair_audit_passed']=numeric['status']=='passed' and numeric['maximum_whole_population_numeric_change']==0
    # Confirm both corrected later analyses retain the same support population.
    rep=ROOT/'production_workflow/training_representation_diagnostic_20260909_a'
    parent=pd.read_csv(rep/'point_diagnostics.csv.gz',usecols=['experiment','configuration','row_id','both_support'])
    child=pd.read_csv(rep/'c_diagnostic/point_c_diagnostics.csv.gz',usecols=list(parent.columns))
    key=['experiment','configuration','row_id']
    pd.testing.assert_frame_equal(parent.set_index(key).sort_index(),child.set_index(key).sort_index(),check_exact=True)
    checks['c_extension_support_identical']=True
    assert all(checks.values()),checks
    # Generate only the self-contained upload package, keeping build files outside it.
    (SOURCE/'manuscript.pdf').write_bytes((BUILD/'manuscript.pdf').read_bytes())
    (BASE/'manuscript.pdf').write_bytes((BUILD/'manuscript.pdf').read_bytes())
    zip_path=BASE/'Testing_point_wise_inference_of_basal_friction_support_corrected_20260910.zip'
    inventory=sorted(p for p in SOURCE.rglob('*') if p.is_file())
    assert all(p.suffix not in {'.aux','.log','.blg','.bbl','.tmp'} for p in inventory)
    with ZipFile(zip_path,'w',ZIP_DEFLATED) as z:
        for p in inventory:z.write(p,p.relative_to(SOURCE).as_posix())
    clean=BUILD/'clean_upload';clean.mkdir(exist_ok=True)
    with ZipFile(zip_path) as z:
        assert z.testzip() is None
        assert set(z.namelist())=={p.relative_to(SOURCE).as_posix() for p in inventory}
        assert all(z.read(p.relative_to(SOURCE).as_posix())==p.read_bytes() for p in inventory)
        z.extractall(clean)
    compiler=ROOT/'latex_revision/tools/tectonic/tectonic.exe'
    subprocess.run([str(compiler),'--only-cached','--keep-logs','--keep-intermediates','manuscript.tex'],cwd=clean,check=True,capture_output=True)
    clean_log=(clean/'manuscript.log').read_text(encoding='utf-8',errors='replace')
    assert not re.search(r'undefined|Overfull|multiply defined',clean_log,re.I)
    assert (clean/'manuscript.aux').read_bytes()==(BUILD/'manuscript.aux').read_bytes()
    checks['clean_upload_compiles_with_identical_references']=True
    checks['zip_crc_and_file_identity']=True
    report={'status':'passed','checks':checks,'figure_files':len(hashes),'references':len(keys),
            'pdf_sha256':sha(SOURCE/'manuscript.pdf'),'zip_sha256':sha(zip_path),
            'zip':str(zip_path),'source_sha256':sha(SOURCE/'manuscript.tex'),
            'known_build_warnings':'existing lineno encoding/fontconfig; one underfull bibliography paragraph',
            'visual_inspection':'All 41 pages reviewed in contact sheets; enlarged affected pages 9, 13 and 40; final page-40 paragraph spacing and line numbers corrected.',
            'no_new_physical_solves_or_training':True}
    (HERE/'release_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
