from pathlib import Path
import zipfile,re,hashlib,json,tempfile
from pypdf import PdfReader
root=Path(__file__).resolve().parent
base=root/"manuscript"
tex=(base/"manuscript.tex").read_text(encoding="utf-8")
files=[base/n for n in ["manuscript.tex","bibliography.bib","igs.cls","igs.bst","README.md","manuscript.pdf"]]
for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]+)\}",tex):
    matches=list(base.rglob(f))
    assert len(matches)==1,(f,matches)
    files.append(matches[0])
assert len(files)==23 and len(set(files))==23
assert 'Provisional:' not in tex and 'confirm the delineation' not in tex
out=root/"JOG_Overleaf_20260916.zip"
with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
    for f in files:z.write(f,f.relative_to(base).as_posix())
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    assert len(z.namelist())==23
    for f in files:assert z.read(f.relative_to(base).as_posix())==f.read_bytes()
    extracted=Path(tempfile.mkdtemp(prefix='clean_check_20260916_',dir=root))
    z.extractall(extracted)
record={"zip_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
        "pdf_sha256":hashlib.sha256((base/"manuscript.pdf").read_bytes()).hexdigest(),
        "files":len(files),"figures":len(files)-6,
        "pages":len(PdfReader(base/'manuscript.pdf').pages),
        "clean_extraction":str(extracted),
        "clean_build_verified":False,
        "source":"author __5_.zip; amendments consistency_updates.docx",
        "google_scholar_export":"inaccessible; verified publication metadata used",
        "backup_updated":False}
(root/"release_record_20260916.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
print(json.dumps(record))
