from pathlib import Path
import subprocess,shutil,zipfile,re,json
R=Path(__file__).resolve().parent; S=R/'source'; B=R.parent/'build'/'paper'
def compile_tex(src,out,name):
 out.mkdir(parents=True,exist_ok=True)
 font_args=['--only-cached','-Z','search-path='+str(R/'build'/'latex_support')] if (R/'build'/'latex_support').exists() else []
 r=subprocess.run(['tectonic']+font_args+['--keep-logs','--keep-intermediates','--outdir',str(out),str(src)],cwd=src.parent,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (out/'compiler_output.txt').write_text(r.stdout)
 if r.returncode: print(r.stdout[-6000:]); raise SystemExit(r.returncode)
 pdf=out/(src.stem+'.pdf'); shutil.copy(pdf,B/name)
 log=(out/(src.stem+'.log')).read_text(errors='replace')
 issues=[x for x in log.splitlines() if any(y in x.lower() for y in ['undefined','overfull','missing character','multiply defined'])]
 return {'pdf':name,'compiler':'Tectonic (XeTeX)','issues':issues}
reports=[]
for name in ['manuscript','supplement']:
 reports.append(compile_tex(S/(name+'.tex'),B/name,name+'.pdf'))
# Combined preprint source, no private audit artifacts.
a=B/'arxiv'; pkg=a/'source';pkg.mkdir(parents=True,exist_ok=True)
for f in S.rglob('*'):
 if f.is_file() and f.suffix in {'.tex','.pdf','.bib'}:
  dest=pkg/f.relative_to(S);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy(f,dest)
wrapper=(S/'manuscript.tex').read_text()
head=wrapper[wrapper.index('\\title{'):wrapper.index('\\begin{document}')]
main=r'''\pdfoutput=1
\documentclass[11pt,a4paper]{article}
\input{preamble.tex}
\newcommand{\supp}[2]{Appendix~\ref{#2}}
'''+head+r'''\begin{document}\maketitle
\input{manuscript_body.tex}
\input{manuscript_references.tex}
\clearpage\appendix
\noindent The appendices report implementation details and complete secondary and exploratory results. F1 and exact match use points ($\times100$); NDCG uses the 0--1 scale. L is condition E in the frozen records.
\input{supplement_body.tex}
\end{document}
'''
(pkg/'main.tex').write_text(main)
check=a/'build';check.mkdir(parents=True,exist_ok=True)
for f in pkg.rglob('*'):
 if f.is_file():
  d=check/f.relative_to(pkg);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy(f,d)
(check/'main.tex').write_text(main.replace('\\pdfoutput=1\n','',1))
reports.append(compile_tex(check/'main.tex',a/'compile','arxiv/main.pdf'))
with zipfile.ZipFile(a/'arxiv_source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in sorted(pkg.rglob('*')):
  if f.is_file():z.write(f,f.relative_to(pkg))
# Anonymous journal candidates, not submitted or represented as accepted replacement files.
j=B/'journal_revision';js=j/'source';js.mkdir(parents=True,exist_ok=True)
for f in S.rglob('*'):
 if f.is_file() and f.suffix in {'.tex','.pdf','.bib'}:
  d=js/f.relative_to(S);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy(f,d)
for name in ['manuscript','supplement']:
 p=js/(name+'.tex');t=p.read_text();start=t.index('\\author{');end=t.index('\\begin{document}')
 t=t[:start]+'\\author{}\n\\date{}\n'+t[end:];p.write_text(t)
 p=js/(name+'_body.tex');t=p.read_text()
 # Remove identifying availability paragraph and CRediT; keep the AI declaration before the references.
 if name=='manuscript':
  t,n=re.subn(r'\\section\*\{Author contributions \(CRediT\)\}.*?(?=\\section\*\{Declaration of generative AI)','',t,flags=re.S); assert n==1
  astart=t.index(r'\paragraph{Data and code availability.}')
  aend=t.index(r'\paragraph{Funding and competing interests.}',astart)
  t=t[:astart]+r'''\paragraph{Data and code availability.} Identifying companion links are supplied separately to the editor. The selected source and aggregate companion does not provide complete model-experiment replay: exact split/selection records, per-record outputs, six final adapters and some executed recovery code are absent, reflecting the chosen release scope. The pre-execution protocol was released after comparative outcomes and is not a public preregistration. Upstream dataset terms and replay dependencies are described in \supp{S7}{supp:repro}.
'''+t[aend:]
  fund=r'\paragraph{Funding and competing interests.} This study was self-funded. The authors declare no competing interests. This manuscript is a preprint and has not been accepted for journal publication.'
  assert t.count(fund)==1
  t=t.replace(fund,r'\paragraph{Funding.} This study was self-funded.'+'\n'+r'\paragraph{Declaration of competing interest.} The authors declare no competing interests.')
 else:
  t=re.sub(r'\\section\*\{Author contributions \(CRediT\)\}.*','',t,flags=re.S)
 t=t.replace('https://github.com/zukhriddingit/chunkrag-reranking-study','https://example.invalid/anonymous-companion')
 t=re.sub(r'The first release \(v1\.0\.0.*?The protocol was fixed', 'Companion access details are supplied separately to the editor. The protocol was fixed',t,flags=re.S)
 t=re.sub(r'A companion with selected source code, aggregate results and the final pre-execution protocol for Study~2 is publicly available at.*?The protocol was fixed', 'A selected code and aggregate companion is identified separately to the editor. The protocol was fixed',t,flags=re.S)
 t=re.sub(r'The public companion provides selected source, configuration, pre-execution protocol and aggregate results\..*?The protocol was fixed', 'A selected source and aggregate companion is identified separately to the editor. The protocol was fixed',t,flags=re.S)
 t=t.replace(r'\url{https://example.invalid/anonymous-companion}', 'the companion (details supplied separately)')
 p.write_text(t)
 p=js/'preamble.tex';t=p.read_text().replace('pdfauthor={Assylkhan Geniyat, Zukhriddin Fakhriddinov, Atharv Gulati}','pdfauthor={}').replace('\\rhead{Preprint---not peer reviewed}','\\rhead{}');p.write_text(t)
 reports.append(compile_tex(js/(name+'.tex'),j/'build'/name,'journal_revision/'+name+'_anonymous.pdf'))
(B/'compilation_report.json').write_text(json.dumps(reports,indent=2)+'\n')
for x in reports:print(x)
