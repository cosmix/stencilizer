from vlib2 import *
def contour_ranges(m):
    r=[];k=0
    for c in m.contours: r.append((k,len(c.points))); k+=len(c.points)
    return r
def replay3(out,plan,m,assign,default_out,K=12):
    mp=[p for c in m.contours for p in c.points]
    rng=contour_ranges(m)
    def owner(i):
        for s,L in rng:
            if s<=i<s+L: return s,L
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
        for q in qs: target[q]=(ax,c,g[1])
    coords=[[None]*len(cp) for cp in plan]
    for ci,cp in enumerate(plan):
        for pi,s in enumerate(cp):
            if s[0]=='v': coords[ci][pi]=[mp[s[1]].x,mp[s[1]].y]
            else:
                A,B,t=mp[s[1]],mp[s[2]],s[3]; coords[ci][pi]=[A.x+t*(B.x-A.x),A.y+t*(B.y-A.y)]
    for (ci,pi),(ax,c,cd) in target.items():
        s=plan[ci][pi]; st,L=owner(s[1]); a0=s[1]-st
        best=None
        for d in sorted(range(-K,K+1),key=abs):
            i=st+(a0+d)%L; j=st+(a0+d+1)%L
            A,B=mp[i],mp[j]; a=(A.x,A.y)[ax]; b=(B.x,B.y)[ax]
            if (a-c)*(b-c)<=0 and abs(b-a)>1e-12:
                t=(c-a)/(b-a); best=(A.x+t*(B.x-A.x),A.y+t*(B.y-A.y)); break
        if best is None: return None
        coords[ci][pi]=list(best); coords[ci][pi][ax]=c
        # project wrong-side neighbours
        dpts=default_out.contours[ci].points; n=len(plan[ci])
        for step in (1,-1):
            k=pi
            while True:
                k=(k+step)%n
                if (ci,k) in target or plan[ci][k][0]!='v': break
                dv=(dpts[k].x,dpts[k].y)[ax]-cd
                mv=coords[ci][k][ax]-c
                if dv==0 or (dv>0)==(mv>0): break
                coords[ci][k][ax]=c
    od=out.to_dict(); cs=[]
    for ci,cd in enumerate(od['contours']):
        cs.append({**cd,'points':[{**pd,'x':coords[ci][pi][0],'y':coords[ci][pi][1]} for pi,pd in enumerate(cd['points'])]})
    od['contours']=cs
    return Glyph.from_dict(od)
