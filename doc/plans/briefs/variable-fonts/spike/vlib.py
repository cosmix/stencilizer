import math
from fontTools.pens.basePen import BasePen
from fontTools.pens.recordingPen import RecordingPen
from stencilizer.io.converter import fonttools_glyph_to_domain
from stencilizer.core.analyzer import GlyphAnalyzer
from stencilizer.core.processor import _transform_glyph
from stencilizer.config.settings import BridgeConfig, GeometryConfig
from stencilizer.domain import Glyph
N=8
class Flat(BasePen):
    def __init__(s,gs): super().__init__(gs); s.rec=RecordingPen(); s.cur=None
    def _moveTo(s,p): s.rec.moveTo(p); s.cur=p
    def _lineTo(s,p): s.rec.lineTo(p); s.cur=p
    def _curveToOne(s,a,b,c):
        p0=s.cur
        for i in range(1,N+1):
            t=i/N; u=1-t
            s.rec.lineTo((u**3*p0[0]+3*u*u*t*a[0]+3*u*t*t*b[0]+t**3*c[0],u**3*p0[1]+3*u*u*t*a[1]+3*u*t*t*b[1]+t**3*c[1]))
        s.cur=c
    def _qCurveToOne(s,a,c):
        p0=s.cur
        for i in range(1,N+1):
            t=i/N; u=1-t
            s.rec.lineTo((u*u*p0[0]+2*u*t*a[0]+t*t*c[0],u*u*p0[1]+2*u*t*a[1]+t*t*c[1]))
        s.cur=c
    def _closePath(s): s.rec.closePath()
    def _endPath(s): s.rec.endPath()
class G:
    def __init__(s,rec): s.rec=rec
    def draw(s,pen): s.rec.replay(pen)
def flat(f,gset,n):
    fp=Flat(gset); gset[n].draw(fp); return fonttools_glyph_to_domain(n,G(fp.rec),f)
def surgery(g,upm):
    return _transform_glyph(g.to_dict(),BridgeConfig().model_dump(),upm,GeometryConfig().model_dump())
def mapping(g,out):
    pts=[p for c in g.contours for p in c.points]
    idx={}
    for i,p in enumerate(pts): idx.setdefault((round(p.x,6),round(p.y,6)),i)
    edges=[];k=0
    for c in g.contours:
        L=len(c.points); edges+= [(k+i,k+(i+1)%L) for i in range(L)]; k+=L
    plan=[]
    for c in out.contours:
        cp=[]
        for p in c.points:
            key=(round(p.x,6),round(p.y,6))
            if key in idx: cp.append(('v',idx[key])); continue
            hit=None
            for a,b in edges:
                A,B=pts[a],pts[b]; dx,dy=B.x-A.x,B.y-A.y; L2=dx*dx+dy*dy
                if L2==0: continue
                t=((p.x-A.x)*dx+(p.y-A.y)*dy)/L2
                if -1e-9<=t<=1+1e-9 and math.hypot(A.x+t*dx-p.x,A.y+t*dy-p.y)<1e-4: hit=('e',a,b,t);break
            if hit is None: return None
            cp.append(hit)
        plan.append(cp)
    return plan
def replay(out,plan,m):
    mp=[p for c in m.contours for p in c.points]
    od=out.to_dict(); cs=[]
    for cd,cp in zip(od['contours'],plan):
        pts=[]
        for pd,s in zip(cd['points'],cp):
            if s[0]=='v': x,y=mp[s[1]].x,mp[s[1]].y
            else: A,B,t=mp[s[1]],mp[s[2]],s[3]; x,y=A.x+t*(B.x-A.x),A.y+t*(B.y-A.y)
            pts.append({**pd,'x':x,'y':y})
        cs.append({**cd,'points':pts})
    od['contours']=cs
    return Glyph.from_dict(od)
