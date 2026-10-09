import sys, itertools
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.io.converter import fonttools_glyph_to_domain
import inspect
path=sys.argv[1]
f=TTFont(path)
upm=f['head'].unitsPerEm
axes=[a.axisTag for a in f['fvar'].axes]
cmap=f.getBestCmap()
chars="ABDOPQRabdegopq04689&@"
an=GlyphAnalyzer()
locs={'default':{}}
for a in axes:
    locs[a+'-']={a:-1.0}; locs[a+'+']={a:1.0}
gss={k:f.getGlyphSet(location=v,normalized=True) for k,v in locs.items()}
sig=inspect.signature(fonttools_glyph_to_domain)
noisl=[];diff=[]
for ch in chars:
    if ord(ch) not in cmap: continue
    n=cmap[ord(ch)]
    res={}
    for k,gs in gss.items():
        g=fonttools_glyph_to_domain(n, gs[n], f)
        h=an.analyze(g,upm)
        res[k]=(len(g.contours),tuple(sorted(h.get_islands())))
    vals=set(res.values())
    if res['default'][1]==(): noisl.append(f"{ch}({res['default'][0]}c)")
    if len(vals)>1: diff.append((ch,res))
print(path.split('/')[-1], 'axes',axes)
print('  no islands at default:',' '.join(noisl))
print('  island set differs across extremes:',diff[:4])
