"""Check rendered page counts, text bounds and object navigation; no inference."""
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/integrated_fyp_20261013'
pdf = OUT / 'integrated_report.pdf'
bbox = OUT / 'rendered_pages/layout_bbox.html'
subprocess.run(['pdftotext', '-bbox', str(pdf), str(bbox)], check=True)
ns = {'x': 'http://www.w3.org/1999/xhtml'}
pages = ET.parse(bbox).getroot().findall('.//x:page', ns)
layout = json.loads((OUT / 'render_validation.json').read_text(encoding='utf-8'))
reader = PdfReader(pdf)
assert len(pages) == len(reader.pages) == layout['total_pages']
assert layout['main_content_pages'] == layout['main_physical_end'] - layout['main_physical_start'] + 1
assert len(layout['table_figure_pages']) == 24
violations = []
for i, page in enumerate(pages, 1):
    width, height = float(page.attrib['width']), float(page.attrib['height'])
    for word in page.findall('x:word', ns):
        a = {k: float(v) for k, v in word.attrib.items()}
        if a['xMin'] < 20 or a['xMax'] > width - 20 or a['yMin'] < 10 or a['yMax'] > height - 10:
            violations.append([i, word.text, a])
assert not violations, violations
text = '\n'.join(page.extract_text() for page in reader.pages)
assert '\ufffd' not in text and '\u25a0' not in text
for label, page in layout['table_figure_pages'].items():
    assert label in reader.pages[page - 1].extract_text(), (label, page)
result = {
    'status': 'PASS', 'total_pages': len(pages),
    'main_pages': layout['main_content_pages'],
    'mapped_tables_and_figures': 24,
    'text_bounds_violations': violations,
    'replacement_or_black_square_characters': 0,
    'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
    'markdown_sha256': hashlib.sha256((OUT / 'integrated_report.md').read_bytes()).hexdigest(),
    'scope': 'Automated text-bound, page-count and object-navigation checks. Visual review is documented separately in completion_checklist.md.'
}
(OUT / 'final_render_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
