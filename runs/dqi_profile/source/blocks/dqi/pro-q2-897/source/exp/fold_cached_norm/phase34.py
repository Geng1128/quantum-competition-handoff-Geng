"""Use the held I6 DAG with a classical mask:34T,21 private, optional zero."""
from exp.jet24.kernel import phase_inverse12_quantum64

def phase_inverse34(b,x,mask,zero_mask=None):
    return phase_inverse12_quantum64(b,x,mask,zero_mask)
