from vlib3 import *
import vlib3
def lines(out,plan,g):
    pts=[p for c in g.contours for p in c.points]
    grp=defaultdict(list)
    for ci,(c,cp) in enumerate(zip(out.contours,plan)):
        for pi,(p,s) in enumerate(zip(c.points,cp)):
            if s[0]=='e':
                A,B=pts[s[1]],pts[s[2]]
                if abs(B.x-A.x)>=abs(B.y-A.y): grp[('x',round(p.x,5))].append((ci,pi))
                else: grp[('y',round(p.y,5))].append((ci,pi))
    return {q:k for k,v in grp.items() if len(v)>=2 for q in v}
