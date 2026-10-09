import sys, math; sys.path.insert(0,sys.argv[2])
from vlib4 import *
from fontTools.ttLib import TTFont
import pathops
f=TTFont(sys.argv[1]); upm=f['head'].unitsPerEm; an=GlyphAnalyzer()
axes=[a.axisTag for a in f['fvar'].axes]
import itertools
locs=[dict(zip(axes,c)) for c in itertools.product((-1.0,0.0,1.0),repeat=len(axes)) if any(c)]
gs=f.getGlyphSet(location={},normalized=True)
def polys(g): return [[(p.x,p.y) for p in c.points] for c in g.contours]
def union(P):
    path=pathops.Path(); pen=path.getPen()
    for c in P:
        pen.moveTo(c[0]); [pen.lineTo(q) for q in c[1:]]; pen.closePath()
    res=pathops.simplify(path,clockwise=True)
    rec=RecordingPen(); res.draw(rec)
    out=[];cur=None
    for op,args in rec.value:
        if op=='moveTo': cur=[args[0]]
        elif op=='lineTo': cur.append(args[0])
        elif op in('closePath','endPath'): out.append(cur); cur=None
        else: raise RuntimeError(op)
    return out
def umap(P,U):
    pts=[q for c in P for q in c]; edges=[];k=0
    for c in P:
        L=len(c); edges+=[(k+i,k+(i+1)%L) for i in range(L)]; k+=L
    idx={}
    for i,q in enumerate(pts): idx.setdefault((round(q[0],4),round(q[1],4)),i)
    plan=[]
    for c in U:
        cp=[]
        for q in c:
            key=(round(q[0],4),round(q[1],4))
            if key in idx: cp.append(('v',idx[key])); continue
            near=[]
            for a,b in edges:
                A,B=pts[a],pts[b]; dx,dy=B[0]-A[0],B[1]-A[1]; L2=dx*dx+dy*dy
                if L2==0: continue
                t=((q[0]-A[0])*dx+(q[1]-A[1])*dy)/L2
                if -1e-6<=t<=1+1e-6 and math.hypot(A[0]+t*dx-q[0],A[1]+t*dy-q[1])<2e-3: near.append((a,b))
            if len(near)!=2: return None
            cp.append(('x',near[0],near[1]))
        plan.append(cp)
    return plan
def ureplay(plan,Pm):
    pts=[q for c in Pm for q in c]; out=[]
    for cp in plan:
        c=[]
        for s in cp:
            if s[0]=='v': c.append(pts[s[1]])
            else:
                (a,b),(cc,d)=s[1],s[2]; A,B,C,D=pts[a],pts[b],pts[cc],pts[d]
                den=(B[0]-A[0])*(D[1]-C[1])-(B[1]-A[1])*(D[0]-C[0])
                if abs(den)<1e-12: return None
                t=((C[0]-A[0])*(D[1]-C[1])-(C[1]-A[1])*(D[0]-C[0]))/den
                c.append((A[0]+t*(B[0]-A[0]),A[1]+t*(B[1]-A[1])))
        out.append(c)
    return out
def toglyph(g,U):
    from stencilizer.domain.contour import Contour, Point
    return Glyph(metadata=g.metadata,contours=[Contour(points=[Point(x,y) for x,y in c]) for c in U])
cmap=f.getBestCmap()
names=[cmap[ord(ch)] for ch in sys.argv[3] if ord(ch) in cmap]
for n in names:
    g=flat(f,gs,n); P=polys(g); U=union(P); plan=umap(P,U)
    if plan is None: print(n,'union unmappable'); continue
    ug=toglyph(g,U); isl=an.analyze(ug,upm).get_islands()
    out,b,u=surgery(ug,upm)
    if b==0: print(n,'contours',len(P),'->',len(U),'islands',isl,'no bridge'); continue
    sp=mapping(ug,out); assign=lines(out,sp,ug); bad=[]
    for loc in locs:
        mg=flat(f,f.getGlyphSet(location=loc,normalized=True),n)
        Um=ureplay(plan,polys(mg))
        if Um is None: bad.append((loc,'degenerate')); continue
        umg=toglyph(g,Um)
        if sp is None: bad.append('surgery unmappable'); break
        r=replay3(out,sp,umg,assign,out)
        if r is None or an.analyze(r,upm).get_islands(): bad.append(loc)
    print(n,'contours',len(P),'->',len(U),'islands',isl,'bridges',b,'unbridged',u,'failing locs',len(bad),'/',len(locs), bad[:2])
