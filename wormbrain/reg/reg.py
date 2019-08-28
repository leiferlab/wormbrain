import wormbrain as wormb

def register(A,B,method="dsmm",**kwargs):
    # Useful for direct access of the registration functions.
    if method=="dsmm":
        return wormb.reg.dsmm(A,B,**kwargs)
