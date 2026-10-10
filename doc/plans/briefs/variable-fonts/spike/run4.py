import sys; sys.path.insert(0,sys.argv[2])
from vlib4 import *
from fontTools.ttLib import TTFont
f=TTFont(sys.argv[1]); upm=f['head'].unitsPerEm; an=GlyphAnalyzer()
axes=[a.axisTag for a in f['fvar'].axes]
locs=[{a:v} for a in axes for v in (-1.0,1.0)]
if len(axes)>1: locs+= [{a:1.0 for a in axes},{a:-1.0 for a in axes}]
gss=[f.getGlyphSet(location=l,normalized=True) for l in locs]
gs=f.getGlyphSet(location={},normalized=True)
st=dict(glyphs=0,unmapped=0,fail_glyphs=0,clamp_glyphs=0); fails=[]
for n in f.getGlyphOrder():
    if n not in gs: continue
    g=flat(f,gs,n)
    if not g.contours or not an.analyze(g,upm).get_islands(): continue
    out,b,u=surgery(g,upm)
    if b==0 or u: continue
    st['glyphs']+=1
    plan=mapping(g,out)
    if plan is None: st['unmapped']+=1; continue
    assign=lines(out,plan,g); bad=0; cl=0
    for gset in gss:
        r=replay3(out,plan,flat(f,gset,n),assign,out)
        if r is None: cl+=1; bad+=1; continue
        if an.analyze(r,upm).get_islands(): bad+=1
    if cl: st['clamp_glyphs']+=1
    if bad: st['fail_glyphs']+=1; fails.append(n)
print(sys.argv[1].split('/')[-1],len(locs),'locations',st,fails[:25])
