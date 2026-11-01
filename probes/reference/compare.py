import numpy as np, json
a=np.load('ane.npz'); o=np.load('official.npz'); ids=json.load(open('ids.json'))
la,lo=a['logits'],o['logits']
print('shape',la.shape,lo.shape,'emb std ane %.4f off %.4f rel %.3f%%'%(a['emb'].std(),o['emb'].std(),100*np.linalg.norm(a['emb']-o['emb'])/np.linalg.norm(o['emb'])))
def cos(x,y): return float((x*y).sum()/np.linalg.norm(x)/np.linalg.norm(y))
print('pos top1agree top5overlap cos KL(off||ane)')
agree=[];kl=[];ov=[];cs=[]
for p in range(len(ids)):
    A,O=la[p],lo[p]
    pa=np.exp(A-A.max());pa/=pa.sum();po=np.exp(O-O.max());po/=po.sum()
    k=float((po*(np.log(po+1e-12)-np.log(pa+1e-12))).sum())
    t5=len(set(np.argsort(-A)[:5])&set(np.argsort(-O)[:5]));ag=A.argmax()==O.argmax();c=cos(A,O)
    agree.append(ag);kl.append(k);ov.append(t5);cs.append(c)
    if p in(0,1,2,5,10,20,30,35):print(p,bool(ag),t5,round(c,4),round(k,4))
print('MEAN top1 %.3f top5 %.2f cos %.4f KL %.4f median KL %.4f'%(np.mean(agree),np.mean(ov),np.mean(cs),np.mean(kl),np.median(kl)))
print('layer rel L2 (ane vs official) at pos 0, 1, 20, 35 | layer')
for L in range(24):
    print(L,[round(100*float(np.linalg.norm(a['trace'][p,L]-o['trace'][p,L])/np.linalg.norm(o['trace'][p,L])),2) for p in(0,1,20,35)], 'std',round(float(a['trace'][35,L].std()),3),round(float(o['trace'][35,L].std()),3))
