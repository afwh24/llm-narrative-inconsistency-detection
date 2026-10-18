"""Display-only cleanup of an existing review Markdown, preserving source files."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re


def readable(text):
    text=html.unescape(text)
    text=re.sub(r'<br\s*/?>','\n',text,flags=re.I)
    # Retain words inside comment-like annotations; only remove their delimiters.
    text=re.sub(r'<!--\s*(.*?)\s*-->',r'\1',text,flags=re.S)
    text=re.sub(r'<!\s*(.*?)\s*>',r'\1',text,flags=re.S)
    labels={'explanation':'Explanation','error_lines':'Error-introducing evidence',
            'contradicted_lines':'Earlier contradicted evidence','decision':'Decision'}
    for tag,label in labels.items():
        text=re.sub(r'<\s*'+tag+r'\s*>','\n\n**'+label+':**\n\n',text,flags=re.I)
        text=re.sub(r'<\s*/\s*'+tag+r'\s*>','\n\n',text,flags=re.I)
    text=re.sub(r'<\s*/?\s*(?:p|li|line|lines|response)\s*>','\n\n',text,flags=re.I)
    text=re.sub(r'<\s*/?\s*(?:m|b|i|em|strong)\s*>','',text,flags=re.I)
    # Non-HTML source annotations are retained visibly, not discarded as tags.
    text=re.sub(r'<([^>\n]+)>',r'[\1]',text)
    return re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+','\n\n',text).strip()


def clean_parsed(value):
    if isinstance(value,str):return readable(value)
    if isinstance(value,list):return [clean_parsed(v) for v in value]
    if isinstance(value,dict):return {k:clean_parsed(v) for k,v in value.items()}
    return value


def transform(original):
    result=[]
    for line in original.splitlines():
        if re.fullmatch(r'<a id="case-\d+"></a>',line):continue
        if line.startswith('## Case '):
            line=re.sub(r'^(## Case \d+) — (.*)$',r'\1\n\n**\2**',line)
        if line.startswith('| Saved response / automatic reference |'):
            continue
        if line=='| --- | --- | --- |':continue
        if line.startswith('| Full raw response |'):
            cells=line.split('|')[2:4]
            for title,value in zip(('Baseline response','verify_revise response'),cells):
                result.extend(['',f'### {title}','',readable(value),''])
            continue
        if line.startswith('| Saved corrected parsed prediction |'):
            cells=line.split('|')[2:4]
            for title,value in zip(('Baseline','verify_revise'),cells):
                parsed=json.loads(html.unescape(value.replace('<br>','\n')))
                result.extend(['',f'#### {title}: saved parsed prediction (display cleaned)','',
                               '```json',json.dumps(clean_parsed(parsed),ensure_ascii=False,indent=2),'```',''])
            continue
        if line.startswith('| AUTOMATIC '):
            cells=line.split('|')[1:4]
            result.extend(['',f'**{cells[0].strip()}**',
                           '- Baseline: '+readable(cells[1]).replace('\n',' '),
                           '- verify_revise: '+readable(cells[2]).replace('\n',' '),''])
            continue
        if '&lt;' in line or '<br>' in line:
            line=readable(line)
        result.append(line)
    result.insert(2,'Reading view: formatting tags are replaced with paragraphs and section labels. Original saved responses, parsed results and CSV annotations are unchanged. Bracketed source annotations retain their wording.\n')
    return '\n'.join(result)+'\n'


def run(path):
    path=Path(path)
    before=path.read_bytes()
    other={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.parent.iterdir() if p.is_file() and p!=path}
    updated=transform(before.decode('utf8'))
    history=path.parent/'readability_history'
    history.mkdir(exist_ok=True)
    backup=history/(path.stem+'_'+hashlib.sha256(before).hexdigest()[:12]+path.suffix)
    if not backup.exists():backup.write_bytes(before)
    path.write_text(updated,encoding='utf8',newline='\n')
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=h for p,h in other.items()):
        raise ValueError('Unexpected change to another review file')
    print(json.dumps({'updated':str(path),'backup':str(backup),'other_files_unchanged':len(other),
                      'cases':updated.count('### Full story'),'raw_response_sections':updated.count('### Baseline response')+updated.count('### verify_revise response')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path)
    run(parser.parse_args().path)
