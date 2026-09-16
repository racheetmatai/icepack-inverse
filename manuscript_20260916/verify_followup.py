"""Check the clean upload build and approved follow-up; no scientific reruns."""
from pathlib import Path
import json, re, subprocess, zipfile
from pypdf import PdfReader
root=Path(__file__).resolve().parent
record_path=root/'release_record_20260916.json'
record=json.loads(record_path.read_text())
extracted=Path(record['clean_extraction'])
out=extracted/'build'; out.mkdir(exist_ok=True)
tectonic=root.parent/'latex_revision/tools/tectonic/tectonic.exe'
subprocess.run([str(tectonic),'-X','compile','manuscript.tex','--outdir',str(out),'--keep-logs'],cwd=extracted,check=True,capture_output=True)
for log in [root/'build/manuscript.log',out/'manuscript.log']:
    assert not re.search(r'undefined|Overfull|Missing character',log.read_text(errors='replace'),re.I)
original=PdfReader(root/'manuscript/manuscript.pdf')
clean=PdfReader(out/'manuscript.pdf')
assert len(original.pages)==len(clean.pages)
assert all(a.get_contents().get_data()==b.get_contents().get_data() for a,b in zip(original.pages,clean.pages))
tex=(root/'manuscript/manuscript.tex').read_text(encoding='utf-8')
with zipfile.ZipFile(root/'JOG_Overleaf_20260915.zip') as z:
    old=z.read('manuscript.tex').decode('utf-8')
    extract=lambda s,env: re.findall(r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}',s,re.S)
    assert extract(old,'equation')==extract(tex,'equation')
    assert extract(old,'tabular')==extract(tex,'tabular')
    conclusion=lambda s:re.split(r'\\section\{Conclusion[s]?\}',s)[1].split('\\section')[0]
    assert conclusion(old)==conclusion(tex)
    old_labels=set(re.findall(r'\\label\{([^}]+)\}',old))
    assert old_labels <= set(re.findall(r'\\label\{([^}]+)\}',tex))
    for name in z.namelist():
        if name.startswith('figures/'):
            assert z.read(name)==(root/'manuscript'/name).read_bytes()
with zipfile.ZipFile(root/'JOG_Overleaf_20260916.zip') as z:
    for name in z.namelist():
        if name.endswith(('.tex','.md')):
            content=z.read(name).decode('utf-8')
            assert not re.search(r'Provisional:|confirm.*Andrew|Andrew.*confirm',content,re.I)
poppler=Path(r'C:\Users\LDEO-GPU-TITANX\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe')
pages=[]
for i,p in enumerate(original.pages,1):
    text=p.extract_text()
    if any(s in text for s in ['The common architecture and training procedure','outlines were delineated','model-selection procedure, and training']):
        pages.append(i)
        subprocess.run([str(poppler),'-f',str(i),'-l',str(i),'-scale-to','1400','-png',str(root/'manuscript/manuscript.pdf'),str(root/'build/final_followup')],check=True,capture_output=True)
record.update(clean_build_verified=True,all_page_content_streams_identical=True,
              equations_tables_conclusion_original_figures_unchanged=True,
              original_labels_preserved=True,reminder_absent_from_upload=True,
              affected_pages_rendered=pages)
record_path.write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record))
