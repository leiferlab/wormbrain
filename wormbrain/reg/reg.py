import wormbrain as wormb

def register(A,B,method="dsmm",**kwargs):
    
    if method=="dsmm":
        return wormb.reg._dsmm(A,B,**kwargs)
