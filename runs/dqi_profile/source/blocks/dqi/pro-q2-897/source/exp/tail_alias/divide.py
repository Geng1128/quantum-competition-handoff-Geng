"""Zero-aware initialized inverse fits below the product peak:369T safe divide."""
from pathlib import Path
import sys
from types import FunctionType
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.fold_joint import divide as prior
from exp.inverse19_joint.kernel import inverse_zero

def inverse_overwrite_zero66(b,x):
    """Overwrite inverse0+zero,66T,31 localQ; old input mask deferred."""
    out,F=inverse_zero(b,x);m=b.alloc_bits(12);b.hmr(x,m);b.swap(x,out);b.free(out)
    return m,F

_globals=dict(prior.divide_zero371.__globals__,inverse_stream_zero=inverse_overwrite_zero66)
_fn=prior.divide_zero371
divide_zero369=FunctionType(_fn.__code__,_globals,'divide_zero369',_fn.__defaults__,_fn.__closure__)
divide_zero369.__doc__='All-input safe GF24 division: <=369 executed T,37 private. Old initialized inverse+flag replaces streamed68T with66T below37-bit product peak.'
