from vlib import *
from collections import defaultdict
def lines(out,plan):
    """group edge points by shared default x (vertical line) or y."""
    ex=defaultdict(list); ey=defaultdict(list)
    for ci,(c,cp) in enumerate(zip(out.contours,plan)):
        for pi,(p,s) in enumerate(zip(c.points,cp)):
            if s[0]=='e': ex[round(p.x,5)].append((ci,pi)); ey[round(p.y,5)].append((ci,pi))
    assign={}
    for k,v in ex.items():
        if len(v)>=2:
            for q in v: assign[q]=('x',k)
    for k,v in ey.items():
        if len(v)>=2:
            for q in v: assign.setdefault(q,('y',k))
    return assign
def replay2(out,plan,m,assign):
    mp=[p for c in m.contours for p in c.points]
    fixed={}
    for ci,cp in enumerate(plan):
        for pi,s in enumerate(cp):
            if s[0]=='e':
                A,B,t=mp[s[1]],mp[s[2]],s[3]; fixed[(ci,pi)]=(A.x+t*(B.x-A.x),A.y+t*(B.y-A.y))
    groups=defaultdict(list)
    for q,g in assign.items(): groups[g].append(q)
    target={}
    for g,qs in groups.items():
        ax=0 if g[0]=='x' else 1
        c=sum(fixed[q][ax] for q in qs)/len(qs)
        for q in qs: target[q]=(ax,c)
    od=out.to_dict(); cs=[]; clamped=0
    for ci,(cd,cp) in enumerate(zip(od['contours'],plan)):
        pts=[]
        for pi,(pd,s) in enumerate(zip(cd['points'],cp)):
            if s[0]=='v': x,y=mp[s[1]].x,mp[s[1]].y
            else:
                A,B,t=mp[s[1]],mp[s[2]],s[3]
                if (ci,pi) in target:
                    ax,c=target[(ci,pi)]; a=(A.x,A.y)[ax]; b=(B.x,B.y)[ax]
                    if abs(b-a)>1e-9:
                        t2=(c-a)/(b-a)
                        if t2<0 or t2>1: clamped+=1; t2=min(1,max(0,t2))
                        t=t2
                x,y=A.x+t*(B.x-A.x),A.y+t*(B.y-A.y)
            pts.append({**pd,'x':x,'y':y})
        cs.append({**cd,'points':pts})
    od['contours']=cs
    return Glyph.from_dict(od),clamped
