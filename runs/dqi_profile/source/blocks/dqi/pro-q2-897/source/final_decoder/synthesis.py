"""Synthesis with additive continuants, shared norms, and no outer fold.

Every imported dependency is an unchanged file in decoderkit.
The caller compensates for the omitted fixed quadratic in the scan.
All arithmetic uses circ builder gates.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent)]
import decode as baseline
from exp.decoder.common import ar
from exp.mul99.schedules import multiply_forward99
from exp.noscratch.schedules import norm_xor_noscratch
from exp.noscratch.kernels import mul_xor_noscratch
from exp.arithmetic.division import phase_product24
from exp.inverse_search.inverse20_iso import inverse6_xor20
from exp.phase24_tradeoff.phase6small import phase_inverse6_classical6
from exp.zero_padding_synthesis.kernel import *
from exp.fold_cached_norm import kernel as FK



def divide_fused(b, x, y, F, fm, after, K=None):
    """Consume F=[y=0] and prior phase fm; divide with identity at zero.

    The callback sees restored y and the live flag.  It may change x, but
    must restore y before returning.  Every norm/flag measurement is closed.
    """
    b.cx(F, y[0])
    b.cx(y[12:], y[:12])
    oldmask = multiply_forward99(b, x, y, K)
    b.cx(y[12:], y[:12])
    N = b.alloc_qubits(12)
    norm_xor_noscratch(b, y, N, K)
    conj = [ar.field(12).frobenius(1 << j, 6) for j in range(12)]
    ar.linear_inplace(b, N, conj)
    masks = []
    for off in (0, 12):
        half = x[off:off + 12]
        out = b.alloc_qubits(12)
        ar.mul_xor0(b, half, N, out)
        m = b.alloc_bits(12)
        b.hmr(half, m)
        b.swap(half, out)
        b.free(out)
        masks.append(m)
    ar.linear_inplace(b, N, conj)
    to, frm, k6 = ar.tower(12)
    ar.linear_inplace(b, N, to)
    M = b.alloc_qubits(6)
    norm_xor_noscratch(b, N, M, k6)
    ar.linear_inplace(b, N, frm)
    t = inverse6_xor20(b, M)
    for off in (0, 12):
        half = x[off:off + 12]
        ar.linear_inplace(b, half, to)
        for j in (0, 6):
            part = half[j:j + 6]
            out = b.alloc_qubits(6)
            mul_xor_noscratch(b, part, t, out)
            ar.erase_product(b, out, M, part)
            b.swap(part, out)
            b.free(out)
        ar.linear_inplace(b, half, frm)
    for off, m in zip((0, 12), masks):
        ar.phase_product(b, x[off:off + 12], N, m)
        b.free(m)
    phase_product24(b, x, y, oldmask, K)
    b.free(oldmask)

    # Norms of a clipped zero factor differ only in their constant bit.
    b.cx(F, y[0])
    b.cx(F, N[0])
    b.cx(F, M[0])
    after()
    fmask = b.alloc_bits(1)
    b.hmr(F, fmask[0])
    b.free(F)
    b.cx(fm[0], fmask[0])
    b.free(fm)
    b.z(fmask[0])
    pm = b.alloc_bits(6)
    b.bit_store(pm, False)
    b.cx(fmask[0], pm[0])
    ar.phase_product(b, M, t, pm)
    b.free(pm)
    b.free(fmask)
    tm = b.alloc_bits(6)
    b.hmr(t, tm)
    b.free(t)
    # t=inv0(M)+[M=0], so its constant output bit also masks the zero phase.
    phase_inverse6_classical6(b, M, tm, zero_mask=tm[0])
    b.free(tm)
    ar.linear_inplace(b, N, to)
    ar._erase_norm(b, N, M, k6)
    b.free(M)
    ar.linear_inplace(b, N, frm)
    ar._erase_norm(b, y, N, K)
    b.free(N)


def absorb_odd_fused(b, head, A, B, Q, p, virtual, pconst, c, kap,
                     reverse=False, transcript=None, wide=False):
    if not reverse:
        return FK.prior.absorb_odd(b, head, A, B, Q, p, virtual, pconst,
                                   c, kap, reverse, transcript, wide)
    k = len(B)
    v = epow(c, k + 2)
    co = [v & 4095, v >> 12]
    R_to_tops(b, head, B, c, True)
    old.add_alias(b, p, head, co)
    value = old.eval_implicit(b, A, Q, c)
    F = dv.zero_flag(b, value)
    hm, fm = transcript
    old.controlled_linear_xor(b, F, p, head, co)
    def after():
        cx(b, head[1], head[0], c)
        old.eval_implicit(b, A, Q, c, True)
        b.cx(F, A[0][0])
        low_product(b, head, A, B)
        b.cx(F, A[0][0])
        q_times_low(b, B, kap, True)
        old.eval_implicit(b, A, Q, c)
    divide_fused(b, combine(head), value, F, fm, after, K)
    b.cz(combine(head), hm)
    b.free(hm)
    old.eval_implicit(b, A, Q, c, True)
    return B + head, A, transcript


def absorb_even_fused(b, head, A, B, Q, r, c, kap,
                      reverse=False, transcript=None):
    if not reverse:
        return FK.prior.absorb_even(b, head, A, B, Q, r, c, kap,
                                    reverse, transcript)
    k = len(B)
    out = head + B
    target = out + [r]
    value = old.eval_implicit(b, A, Q, c)
    F = dv.zero_flag(b, value)
    old.eval_implicit(b, A, Q, c, True)
    old.controlled_linear_xor(b, F, r, out[-2:], [kap, 1])
    linear_words(b, target, base.embedding_matrix(k + 3, kap), True)
    value = old.eval_implicit(b, A, Q, c)
    cx(b, head[1], head[0], c)
    hm, fm = transcript
    def after():
        cx(b, head[1], head[0], c)
        old.eval_implicit(b, A, Q, c, True)
        full_quotient(b, head, A, Q, B, r, kap)
        old.eval_implicit(b, A, Q, c)
    divide_fused(b, combine(head), value, F, fm, after, K)
    b.cz(combine(head), hm)
    b.free(hm)
    old.eval_implicit(b, A, Q, c, True)
    return out, A, transcript


# Bind only the two inner fold reverses into the original private graph.
_shared_unsynthesize = FK.bind(baseline.unsynthesize,
    absorb=absorb_odd_fused, absorb_even=absorb_even_fused)

sg = baseline.synthesize.__globals__
ug = _shared_unsynthesize.__globals__
ar = sg['ar']


def add_tail(b, head, q, d, odd):
    a, x = head
    if not odd:
        b.cx(x, a)
    sg['add_tail_a'](b, a, q, d)
    if not odd:
        b.cx(x, a)
