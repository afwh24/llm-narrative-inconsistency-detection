"""Render the integrated Markdown and deterministic saved-evidence figures.

Uses ReportLab and Poppler; never calls models or writes experimental sources.
"""
from pathlib import Path
import json, re, html, subprocess, shutil
from reportlab.pdfgen import canvas
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, KeepTogether
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/integrated_fyp_20261013'
FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
QA=OUT/'rendered_pages'; QA.mkdir(exist_ok=True)
A=json.loads((OUT/'verified_aggregates.json').read_text(encoding='utf-8'))
for name,filename in [('Body','times.ttf'),('BodyBold','timesbd.ttf'),('BodyItalic','timesi.ttf'),('Sans','arial.ttf'),('SansBold','arialbd.ttf')]:
    pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+filename))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBold')
BLUE=colors.HexColor('#244b73'); ORANGE=colors.HexColor('#b45f19'); GREY=colors.HexColor('#66717c')

def figure(name,w,h,draw):
    p=FIG/(name+'.pdf'); c=canvas.Canvas(str(p),pagesize=(w,h)); draw(c); c.save()
    subprocess.run(['pdftoppm','-singlefile','-scale-to','2000','-png',str(p),str(FIG/name)],check=True,capture_output=True)
def label(c,x,y,text,size=10,font='Sans',color=colors.black):
    c.setFillColor(color); c.setFont(font,size); c.drawString(x,y,str(text))
def line(c,x1,y1,x2,y2,color=GREY,width=1):
    c.setStrokeColor(color); c.setLineWidth(width); c.line(x1,y1,x2,y2)
def rect(c,x,y,w,h,color): c.setFillColor(color); c.rect(x,y,w,h,stroke=0,fill=1)
def design(c):
    for y,title,sub in [(235,'Official benchmark','100 stories × 6 conditions'),(125,'Generated candidates','30 pairs across 10 writing prompts')]:
        rect(c,20,y-30,225,70,colors.HexColor('#edf2f6')); label(c,30,y+15,title,13,'SansBold'); label(c,30,y-7,sub,11)
    rect(c,315,205,240,70,colors.HexColor('#edf2f6')); label(c,325,250,'Detection and localisation',13,'SansBold'); label(c,325,229,'600 final responses',11)
    rect(c,315,100,240,70,colors.HexColor('#edf2f6')); label(c,325,145,'Human eligibility review',13,'SansBold'); label(c,325,124,'1 accepted / 22 rejected / 7 unresolved',10)
    line(c,245,238,309,238,BLUE,2); line(c,245,135,309,135,BLUE,2)
    label(c,20,58,'Selected explanation review',12,'SansBold'); label(c,20,40,'30 model/story pairs; 60 judgments',10)
    label(c,315,58,'Supplementary detection',12,'SansBold'); label(c,315,40,'1 accepted pair × 6 conditions = 12 records',10)
    line(c,427,100,427,72,BLUE,2); line(c,20,85,290,85,colors.HexColor('#d9d9d9'))
    label(c,20,10,'Gold labels, pair mapping and provenance are used only for evaluation.',10,color=GREY)
figure('study_design',580,300,design)

def scatter(c):
    x0,y0,scale=64,44,3.2
    for v in [0,25,50,75,100]:
        x=x0+v*scale; y=y0+v*2.0
        line(c,x,y0,x,y0+200,colors.HexColor('#dddddd'),0.4)
        line(c,x0,y,x0+320,y,colors.HexColor('#dddddd'),0.4)
        label(c,x-7,y0-17,v,9); label(c,x0-27,y-3,v,9)
    label(c,132,7,'False-positive rate (%)',11); label(c,64,268,'Recall (%)',11)
    for i,(stem,name,col) in enumerate([('qwen3_8b','Qwen 8B',BLUE),('llama3_1_8b','Llama 8B',ORANGE),('mistral_7b','Mistral 7B',colors.HexColor('#466b47'))]):
        ms=[next(x for x in A['main'] if x['condition']==stem+'_'+s) for s in ['baseline','verify_revise']]
        pts=[(x0+m['false_positive_rate']*320,y0+m['recall']*200) for m in ms]
        line(c,*pts[0],*pts[1],col,1.6)
        c.setStrokeColor(col); c.setFillColor(colors.white); c.circle(*pts[0],4,stroke=1,fill=1)
        c.setFillColor(col); c.circle(*pts[1],4,stroke=1,fill=1)
        dx=pts[1][0]-pts[0][0]; dy=pts[1][1]-pts[0][1]; norm=(dx*dx+dy*dy)**.5
        ux,uy=dx/norm,dy/norm
        line(c,*pts[1],pts[1][0]-9*ux+4*uy,pts[1][1]-9*uy-4*ux,col,1.5)
        line(c,*pts[1],pts[1][0]-9*ux-4*uy,pts[1][1]-9*uy+4*ux,col,1.5)
        label(c,415,220-i*36,name,11,'SansBold',col)
    label(c,415,93,'Open: baseline',10); label(c,415,74,'Filled: verify-and-revise',10); label(c,415,45,'50 positives',10); label(c,415,27,'50 negatives',10)
figure('recall_false_alarms',580,290,scatter)

def localization(c):
    x0=180; width=330
    for i,m in enumerate(A['main']):
        y=226-i*33; short=m['condition'].replace('qwen3_8b','Qwen').replace('llama3_1_8b','Llama').replace('mistral_7b','Mistral').replace('_verify_revise',' strategy').replace('_baseline',' baseline')
        label(c,12,y+3,short,11)
        tn=m['tn']; loc=round(m['ceeval_pos']*50)
        rect(c,x0,y,tn/100*width,16,colors.HexColor('#abbac8'))
        rect(c,x0+tn/100*width,y,loc/100*width,16,BLUE)
        label(c,x0+(tn+loc)/100*width+7,y+3,f'{tn} + {loc}',10)
    line(c,x0,45,x0+width,45)
    for n in [0,25,50,75,100]: label(c,x0+n/100*width-4,30,n,9)
    rect(c,15,9,12,12,colors.HexColor('#abbac8')); label(c,33,10,'Correct negatives',10)
    rect(c,230,9,12,12,BLUE); label(c,248,10,'Localised positives',10)
    label(c,180,264,'CEEval-Full numerator out of 100',12,'SansBold')
figure('localisation_decomposition',580,288,localization)

def gen(c):
    x0=205; width=320
    rows=[('Original complete',11,BLUE),('After recovery v2',20,BLUE),('Final technical complete',30,BLUE),('Accepted',1,colors.HexColor('#466b47')),('Rejected',22,ORANGE),('Unresolved',7,GREY)]
    for i,(name,n,col) in enumerate(rows):
        y=232-i*35; label(c,12,y+3,name,11); rect(c,x0,y,n/30*width,17,col); label(c,x0+n/30*width+7,y+3,f'{n}/30',10)
    label(c,205,273,'Scheduled candidate pairs',12,'SansBold')
    line(c,x0,41,x0+width,41)
    for n in [0,10,20,30]:label(c,x0+n/30*width-4,25,n,9)
figure('generation_outcomes',590,290,gen)

W,H=A4; M=64; TW=W-2*M
styles={
 'body':ParagraphStyle('body',fontName='Body',fontSize=12,leading=16.5,spaceAfter=6,alignment=TA_JUSTIFY,splitLongWords=True),
 'h1':ParagraphStyle('h1',fontName='BodyBold',fontSize=18,leading=22,spaceAfter=18,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='BodyBold',fontSize=13,leading=17,spaceBefore=14,spaceAfter=7,keepWithNext=True),
 'title':ParagraphStyle('title',fontName='BodyBold',fontSize=24,leading=31,spaceBefore=60,spaceAfter=28),
 'caption':ParagraphStyle('caption',fontName='BodyItalic',fontSize=10.5,leading=13,spaceBefore=9,spaceAfter=7,keepWithNext=True),
 'figcaption':ParagraphStyle('figcaption',fontName='BodyItalic',fontSize=10.5,leading=13,spaceAfter=12),
 'cell':ParagraphStyle('cell',fontName='Body',fontSize=9.5,leading=11.5),
 'headcell':ParagraphStyle('headcell',fontName='BodyBold',fontSize=9.5,leading=11.5),
 'code':ParagraphStyle('code',fontName='Sans',fontSize=8.5,leading=12,spaceAfter=5),
 'list':ParagraphStyle('list',fontName='Body',fontSize=10.5,leading=13,spaceAfter=6),
}
def fmt(s):
    s=html.escape(s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)',r'<i>\1</i>',s)
    s=re.sub(r'`(.+?)`',r'<font name="Sans" size="10">\1</font>',s)
    s=re.sub(r'(https?://[^\s]+)',r'<link href="\1" color="#244b73">\1</link>',s)
    return s

class Report(BaseDocTemplate):
    def __init__(self,path):
        super().__init__(str(path),pagesize=A4,leftMargin=M,rightMargin=M,topMargin=58,bottomMargin=55,title='Detecting and Explaining Narrative Inconsistencies with Local Large Language Models',author='FYP integrated draft')
        self.addPageTemplates(PageTemplate('normal',frames=[Frame(M,55,TW,H-113,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=self.footer))
        self.chapter_pages={}; self.heading_pages={}; self.object_pages={}
    def beforeDocument(self): self.chapter_pages={}; self.heading_pages={}; self.object_pages={}
    def footer(self,c,d):
        if d.page==1:return
        c.setFont('Body',9); c.setFillColor(GREY)
        c.drawString(M,H-32,'Local language models and narrative inconsistency')
        c.drawRightString(W-M,30,str(d.page))
    def afterFlowable(self,f):
        if isinstance(f,Paragraph) and hasattr(f,'toc_level'):
            t=f.getPlainText(); key='h'+str(len(self.heading_pages)); self.canv.bookmarkPage(key)
            self.notify('TOCEntry',(f.toc_level,t,self.page,key)); self.heading_pages[t]=self.page
            if f.toc_level==0:self.chapter_pages[t]=self.page
        if isinstance(f,Paragraph) and hasattr(f,'obj_label'):
            self.object_pages[f.obj_label]=self.page
            if hasattr(f,'list_text'):self.notify('ListEntry',(0,f.list_text,self.page))

class ObjectList(TableOfContents):
    def notify(self,kind,stuff):
        if kind=='ListEntry':super().notify('TOCEntry',stuff)

source=(OUT/'integrated_report.md').read_text(encoding='utf-8')
source=re.sub(r'\n<!-- BEGIN GENERATED NAV -->.*?<!-- END GENERATED NAV -->\n','\n',source,flags=re.S)
ls=source.splitlines(); story=[]; i=0
toc=TableOfContents(); toc.levelStyles=[ParagraphStyle('toc0',fontName='BodyBold',fontSize=11,leading=15,spaceBefore=7),ParagraphStyle('toc1',fontName='Body',fontSize=10,leading=13,leftIndent=12,spaceBefore=2)]
objects=ObjectList();objects.levelStyles=[ParagraphStyle('obj',fontName='Body',fontSize=10,leading=12,spaceBefore=1)]
captions=[x for x in ls if re.match(r'^Table [A-Z0-9]+\.\d+\.\s',x)]
figcaptions=[re.match(r'!\[(.*?)\]',x).group(1) for x in ls if x.startswith('![')]
while i<len(ls):
    s=ls[i].strip(); i+=1
    if not s:continue
    if s=='<!-- TOC -->':story.append(toc);continue
    if s=='<!-- LISTS -->':
        story.append(objects)
        continue
    if s.startswith('# '):
        title=s[2:]
        if i==1:story.append(Paragraph(fmt(title),styles['title']))
        else:
            story.append(PageBreak()); p=Paragraph(fmt(title),styles['h1']); p.toc_level=0; story.append(p)
        continue
    if s.startswith('## '):
        title=s[3:]
        if title in ['Abstract','Report status and reading guide','Contents','List of tables and figures']:story.append(PageBreak())
        p=Paragraph(fmt(title),styles['h2'])
        if re.match(r'\d+\.',title):p.toc_level=1
        story.append(p);continue
    if s.startswith('!['):
        match=re.match(r'!\[(.*?)\]\((.*?)\)',s); cap,path=match.groups()
        im=Image(str(OUT/path)); im.drawHeight=TW*im.imageHeight/im.imageWidth; im.drawWidth=TW
        p=Paragraph(fmt(cap),styles['figcaption']);p.obj_label=re.match(r'(Figure \d+\.\d+)',cap).group(1);p.list_text=cap.split('. ',1)[0]+'. '+cap.split('. ',1)[1].split('. ')[0]
        story.append(KeepTogether([im,p]));continue
    if s.startswith('|'):
        rows=[s]
        while i<len(ls) and ls[i].strip().startswith('|'):rows.append(ls[i].strip());i+=1
        cells=[[c.strip() for c in row.strip('|').split('|')] for row in rows]
        cells=[r for r in cells if not all(re.fullmatch(r'[-: ]+',c) for c in r)]
        n=len(cells[0]); first=1.8 if n>=5 else 1.25
        widths=[TW*first/(n-1+first)]+[TW/(n-1+first)]*(n-1)
        if n==2: widths=[TW*.38,TW*.62]
        parsed=[]
        for ri,row in enumerate(cells):
            built=[]
            for ci,c in enumerate(row):
                st=styles['headcell' if ri==0 else 'cell']
                if ci>0 and re.fullmatch(r'[\d\s.,+/%→−-]+',c):st=ParagraphStyle('numeric',parent=st,alignment=TA_CENTER)
                built.append(Paragraph(fmt(c),st))
            parsed.append(built)
        t=Table(parsed,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7edf2')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#d9d9d9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([t,Spacer(1,10)]);continue
    if s.startswith('```'):
        while i<len(ls) and not ls[i].startswith('```'):
            story.append(Paragraph(html.escape(ls[i]),styles['code']));i+=1
        i+=1;continue
    if re.match(r'^Table [A-Z0-9]+\.\d+\.\s',s):
        p=Paragraph(fmt(s),styles['caption']);p.obj_label=s.split('. ',1)[0];p.list_text=s.split('. ',1)[0]+'. '+s.split('. ',1)[1].split('. ')[0];story.append(p);continue
    para=[s]
    while i<len(ls) and ls[i].strip() and not ls[i].startswith(('#','|','![','```','<!--')):
        para.append(ls[i].strip());i+=1
    story.append(Paragraph(fmt(' '.join(para)),styles['body']))

doc=Report(OUT/'integrated_report.pdf'); doc.multiBuild(story)
reader=PdfReader(OUT/'integrated_report.pdf')
start=doc.chapter_pages['1 Introduction']; ref=doc.chapter_pages['References']
nav=[]
for heading,page in doc.heading_pages.items():
    slug=re.sub(r'[^a-z0-9 -]','',heading.lower()).replace(' ','-')
    nav.append(f'- [{heading}](#{slug}) — PDF page {page}')
objnav=[f'- {label} — PDF page {page}' for label,page in doc.object_pages.items()]
final_source=source.replace('<!-- TOC -->','<!-- TOC -->\n<!-- BEGIN GENERATED NAV -->\n'+'\n'.join(nav)+'\n<!-- END GENERATED NAV -->\n').replace('<!-- LISTS -->','<!-- LISTS -->\n<!-- BEGIN GENERATED NAV -->\n'+'\n'.join(objnav)+'\n<!-- END GENERATED NAV -->\n')
(OUT/'integrated_report.md').write_text(final_source,encoding='utf-8')
report={'total_pages':len(reader.pages),'main_content_pages':ref-start,'main_physical_start':start,'main_physical_end':ref-1,
 'preliminary_pages':start-1,'references_and_appendices_pages':len(reader.pages)-ref+1,
 'chapter_start_pages':doc.chapter_pages,'heading_pages':doc.heading_pages,'table_figure_pages':doc.object_pages,
 'body_font':'Times New Roman 12 pt','body_leading_pt':16.5,'page_size':'A4','side_margins_pt':M,
 'main_words':len(source.split('# 1 Introduction')[1].split('# References')[0].split()),'total_words':len(source.split())}
(OUT/'render_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
subprocess.run(['pdftoppm','-r','90','-png',str(OUT/'integrated_report.pdf'),str(QA/'page')],check=True,capture_output=True)
from PIL import Image as PILImage, ImageOps, ImageDraw
files=sorted(QA.glob('page-*.png'))[:len(reader.pages)]
for batch in range(0,len(files),12):
    contact=PILImage.new('RGB',(960,1080),'#d0d0d0'); draw=ImageDraw.Draw(contact)
    for j,p in enumerate(files[batch:batch+12]):
        im=PILImage.open(p).convert('RGB'); im.thumbnail((230,325)); x=(j%4)*240; y=(j//4)*360
        contact.paste(im,(x+(230-im.width)//2,y+22)); draw.text((x+10,y+4),p.stem,fill='black')
    contact.save(QA/f'contact-{batch//12+1}.jpg')
print(json.dumps({k:v for k,v in report.items() if k not in ['heading_pages','table_figure_pages']},indent=2))
