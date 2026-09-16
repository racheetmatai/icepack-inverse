"""Draw the five-step experimental overview; no numerical analysis."""
import os
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

root = Path(__file__).resolve().parents[1]
out = Path(os.environ.get(
    "JOG_METHOD_OVERVIEW_OUTPUT",
    root / "manuscript/figures/appendix/method_overview.pdf",
))
out.parent.mkdir(parents=True, exist_ok=True)
c = canvas.Canvas(str(out), pagesize=(540, 256))
c.setTitle('From reference basal-friction control to withheld-region velocity evaluation')
steps = [
    ('Obtain reference C by inversion',
     ['Fit the ice-flow model to observed velocity across the sector.',
      'This includes observations inside the later test regions.']),
    ('Withhold a region from MLP training and validation',
     ['For square tests, exclude the central square and its buffer.',
      'For regional tests, exclude the selected region.']),
    ('Train MLPs to predict C from local predictors',
     ['Use reference C as the target outside the excluded region.',
      'Do not supply observed velocity as an MLP predictor.']),
    ('Combine predicted C and simulate velocity',
     ['Take the median of ten predictions at each mesh vertex.',
      'Use this C field in one Icepack simulation.']),
    ('Evaluate velocity in the withheld region',
     ['Compare modeled velocity with observed velocity.',
      'Also evaluate the uniform-C and inversion references.']),
]
heading = ParagraphStyle('heading', fontName='Helvetica-Bold', fontSize=11,
                         leading=13, textColor=HexColor('#203b50'))
body = ParagraphStyle('body', fontName='Helvetica', fontSize=10.5,
                      leading=13, textColor=HexColor('#222222'))
for i, (title, lines) in enumerate(steps):
    left = 5 + 108*i
    top = 251
    bottom = 5
    c.setFillColor(HexColor('#f3f6f8'))
    c.setStrokeColor(HexColor('#526879'))
    c.setLineWidth(0.8)
    c.roundRect(left, bottom, 98, 246, 5, fill=1, stroke=1)
    c.setFillColor(HexColor('#203b50'))
    c.setFont('Helvetica-Bold', 12)
    c.drawString(left+8, top-18, str(i+1))
    para = Paragraph(title, heading)
    _, h = para.wrap(82, 90)
    para.drawOn(c, left+8, top-28-h)
    y = top-100
    for line in lines:
        para = Paragraph(line, body)
        _, h = para.wrap(82, 180)
        y -= h
        assert y >= bottom+8, (i, y)
        para.drawOn(c, left+8, y)
        y -= 9
    if i < 4:
        c.setStrokeColor(HexColor('#526879'))
        c.setFillColor(HexColor('#526879'))
        c.line(left+99, 128, left+103, 128)
        p=c.beginPath(); p.moveTo(left+103,131); p.lineTo(left+107,128)
        p.lineTo(left+103,125); p.close()
        c.drawPath(p, fill=1, stroke=0)
c.save()
print(out)
