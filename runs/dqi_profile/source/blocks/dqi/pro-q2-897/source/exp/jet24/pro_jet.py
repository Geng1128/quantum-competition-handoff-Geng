"""Complete jet kernel:269 forward+389 reverse=658 Toffolis,43 private qubits.
Only supplied-library imports and the documented builder API are used.
Tokens contain only classical bits; no per-call state survives elsewhere.
Static masks are little endian, with affine bit0=constant; linear maps list columns.
Plans r/g/p change and retire a source basis, grow outputs, or add one product.
Every retired function's exact quadratic measurement phase is repaid immediately.
"""
from field import bits, field, extension_columns, linear_inplace, square_columns, inverse_columns, tower
from exp.mul99.kernels import mul_xor99_zero, phase_product_quantum99
from exp.noscratch.schedules import norm_xor_noscratch
from exp.noscratch.kernels import rankone_xor, mul_xor_noscratch
from field import tensor, apply, linear_xor
from exp.jet24.kernel import safe_square
import field as ar
from exp.inverse_search import inverse20 as ref
LEAF_OLD12 = ([3334, 28416, 16128, 9744], [[86, 118], [84, 112], [76, 96], [70, 126], [64, 122]], 7, 10)
LEAF1 = ([15939726, 1292428, 48148704, 42382372, 51177556, 58773552], [[130, 530], [326, 658], [474, 542], [220, 840], [456, 664], [176, 304], [318, 512], [310, 690], [236, 776], [394, 638]], 10, 16)

def forward_canonical(b, u, v):
    z, F = inverse24(b, u)
    mu = b.alloc_bits(24)
    b.hmr(u, mu)
    b.swap(u, z)
    b.free(z)
    out = b.alloc_qubits(24)
    safe_square(b, u, F)
    with b.mark('multiply v'): mul_xor99_zero(b, v, u, out)
    safe_square(b, u, F, True)
    mv = b.alloc_bits(24)
    b.hmr(v, mv)
    b.swap(v, out)
    b.free(out)
    fz = b.alloc_bits(1)
    b.hmr(F, fz[0])
    b.free(F)
    return (mu, mv, fz)

def scale_halves(b, x, t, N):
    """Scale each half by nonzero t; N is its multiplicative inverse."""
    for offset in (0, 12):
        half = x[offset:offset + 12]
        out = b.alloc_qubits(12)
        mul_xor_noscratch(b, half, t, out)
        ar.erase_product(b, out, N, half)
        b.swap(half, out)
        b.free(out)

def erase_inverse_cached(b, t, N, k, F, fz):
    """Erase N=inv0(t), k=Norm(t), F=[t=0]; include forward flag debt."""
    to, frm, K = tower(12)
    linear_inplace(b, t, to)
    linear_inplace(b, N, to)
    m = b.alloc_bits(12)
    b.hmr(N, m)
    b.free(N)
    L = b.alloc_qubits(6)
    f = field(6)

    def coeff():
        for j in range(6):
            for i in range(6):
                for r in bits(f.mul(1 << i, 1 << j)):
                    b.ccx(m[r], t[i], L[j])
                    b.ccx(m[r], t[i + 6], L[j])
                    b.ccx(m[r + 6], t[i + 6], L[j])
    coeff()
    phase_inverse6_heldflag17(b, k, L, F, fz[0])
    coeff()
    b.free(L)
    b.free(m)
    b.free(fz)
    ar._erase_norm(b, t, k, K)
    b.free(k)
    linear_inplace(b, t, frm)

def start_inverse(b, N):
    """Keep N; return k=inv0(Norm(N)) and F=[N=0]."""
    to, frm, K = tower(12)
    linear_inplace(b, N, to)
    D = b.alloc_qubits(6)
    norm_xor_noscratch(b, N, D, K)
    k, F = inverse6_zero18(b, D)
    ar._erase_norm(b, N, D, K)
    b.free(D)
    linear_inplace(b, N, frm)
    return (k, F)

def finish_inverse(b, N, k):
    """For safe nonzero N, finish t=inv(N), retaining k=Norm(t)."""
    to, frm, K = tower(12)
    linear_inplace(b, N, to)
    t = b.alloc_qubits(12)
    b.cx(N[6:], N[:6])
    mul_xor_noscratch(b, N[:6], k, t[:6])
    mul_xor_noscratch(b, N[6:], k, t[6:])
    b.cx(N[6:], N[:6])
    linear_inplace(b, N, frm)
    linear_inplace(b, t, frm)
    return t

def reverse_canonical(b, u, v, token):
    mu, mv, fz = token
    to, frm, K = tower(24)
    linear_inplace(b, u, to)
    N = b.alloc_qubits(12)
    with b.mark('reverse norm'): norm_xor_noscratch(b, u, N, K)
    with b.mark('reverse start inverse with flag'): k, F = start_inverse(b, N)
    b.cx(u[12:], u[:12])
    linear_inplace(b, u, frm)
    safe_square(b, u, F)
    out = b.alloc_qubits(24)
    with b.mark('reverse multiply v'): mul_xor99_zero(b, v, u, out)
    safe_square(b, u, F, True)
    linear_inplace(b, u, to)
    b.cx(u[12:], u[:12])
    vm = b.alloc_bits(24)
    b.hmr(v, vm)
    b.swap(v, out)
    b.free(out)
    b.cx(F, N[0])
    b.cx(F, k[0])
    with b.mark('reverse finish inverse'): t = finish_inverse(b, N, k)
    linear_inplace(b, v, to)
    sq = square_columns(12)
    linear_inplace(b, t, sq)
    linear_inplace(b, N, sq)
    with b.mark('reverse scale v'): scale_halves(b, v, t, N)
    linear_inplace(b, N, inverse_columns(sq))
    linear_inplace(b, t, inverse_columns(sq))
    linear_inplace(b, v, frm)
    linear_inplace(b, u, frm)
    safe_square(b, u, F)
    phase_product_quantum99(b, v, u, vm)
    safe_square(b, u, F, True)
    b.free(vm)
    linear_inplace(b, u, to)
    b.cx(F, u[0])
    b.cx(u[12:], u[:12])
    with b.mark('reverse scale u'): scale_halves(b, u, t, N)
    b.cx(F, u[0])
    b.cx(F, N[0])
    b.cx(F, t[0])
    b.cx(F, k[0])
    with b.mark('reverse cached phase inverse'): erase_inverse_cached(b, t, N, k, F, fz)
    ar._erase_norm(b, u, t, K)
    b.free(t)
    linear_inplace(b, u, frm)
    b.cz(u, mu)
    b.cz(v, mv)
    b.free(mu)
    b.free(mv)

def jet_forward(b, u, v):
    """Apply J in the decoder basis; return classical (mu, mv, fz)."""
    cols, frm, _ = extension_columns(None)
    linear_inplace(b, u, cols)
    linear_inplace(b, v, cols)
    token = forward_canonical(b, u, v)
    linear_inplace(b, u, frm)
    linear_inplace(b, v, frm)
    return token

def jet_reverse(b, u, v, token):
    """Restore the input and repay every deferred measurement sign."""
    cols, frm, _ = extension_columns(None)
    linear_inplace(b, u, cols)
    linear_inplace(b, v, cols)
    reverse_canonical(b, u, v, token)
    linear_inplace(b, u, frm)
    linear_inplace(b, v, frm)

def inverse6_zero18_plan(b, x):
    return inverse6_fixed18(b, x, True)
PLAN12 = {'n': 12, 'terms': [(33, 2112, 35), (56, 3584, 905), (25, 1600, 362), (55, 3520, 3), (32, 2048, 334), (23, 1472, 397), (21, 1344, 40), (6, 384, 918), (19, 1216, 190), (3, 192, 50), (30, 1920, 656), (29, 1856, 34), (48, 3072, 18), (28, 1792, 87), (44, 2816, 197)], 'affine': [1, 4, 16, 3, 12, 48, 63, 57, 33, 2, 8, 32], 'reduce': [65, 129, 3, 256, 1, 512, 4, 8, 26, 55], 'retire': [24619, 15261, 374, 2885], 'out': [3216, 2004, 529, 338, 206, 123, 1, 2, 4, 8, 16, 32], 'size': 10, 'leaves': [([33, 56, 2112, 3584, 135168, 229376], [9, 42, 585, 2730], LEAF_OLD12), ([55, 32, 3520, 2048, 225280, 131072], [14, 13, 910, 845], LEAF_OLD12), ([21, 6, 1344, 384, 86016, 24576], [22, 62, 1430, 4030], LEAF_OLD12), ([3, 30, 192, 1920, 12288, 122880], [16, 34, 1040, 2210], LEAF_OLD12), ([48, 28, 3072, 1792, 196608, 114688], [23, 5, 1495, 325], LEAF_OLD12)]}
PLAN24 = {'n': 24, 'terms': [(1776, 1114384, 419755), (1969, 6951184, 1511959), (1697, 332048, 3917), (321, 8257536, 1627580), (2016, 8065296, 1628913), (272, 1315137, 3418), (592, 15793935, 1270639), (3439, 2494223, 177757), (608, 200463, 2329), (3903, 14020608, 628018), (3423, 14089999, 629803), (3855, 15990591, 3908), (3726, 9349354, 1770621), (1207, 12966122, 538163), (3165, 2963690, 1869), (2617, 6701056, 1626702), (1636, 4946154, 1627395), (2282, 10721849, 382), (1877, 9365742, 1084187), (3879, 8165614, 1481953), (1993, 641262, 1299), (2162, 16494592, 604154), (4027, 15890670, 602857), (2286, 8857714, 2546), (119, 2089470, 1642329), (714, 3359230, 238442), (820, 3420670, 1033), (701, 1609728, 39987), (393, 2925054, 38970), (510, 2871997, 1891), (2300, 16154471, 1179523), (233, 16314215, 1165379), (3982, 7810919, 412), (2069, 7974912, 903104), (1947, 958311, 902748), (3943, 8476693, 2527), (888, 8947848, 1473400), (984, 11864200, 1664433), (2896, 8554632, 415), (160, 12517376, 917193), (3056, 4032648, 917334), (2184, 655520, 1070)], 'affine': [755, 2929, 2347, 572, 2191, 2631, 3604, 1538, 3661, 1633, 237, 499, 3208, 725, 2900, 3574, 1837, 3303, 873, 3492, 1637, 2503, 1978, 3771], 'reduce': [1, 2, 4, 9, 22, 61, 127, 251, 394, 632, 2036, 3457, 4096, 12288, 20480, 61440, 77824, 143360, 286720, 524288, 1056768], 'retire': [], 'out': [4910144, 9817152, 3062784, 1540096, 111680, 186368, 266304, 535616, 46080, 25600, 7232, 10304, 65, 1027, 2053, 15, 18, 34, 134, 256, 515, 1, 2, 4], 'size': 21, 'leaves': [([321, 272, 2016, 8257536, 6950912, 1114112, 4563402752, 29796335616, 28470935552], [1468, 2801, 3418, 9332966, 6014396, 11475697], LEAF1), ([3903, 3855, 3423, 14020608, 2490368, 15790080, 64676167680, 9932111872, 10200547328], [1330, 3115, 3908, 10971766, 5449010, 12762155], LEAF1), ([2617, 2282, 1636, 6701056, 12963840, 9347072, 38285606912, 62511906816, 53099888640], [590, 1283, 382, 3343152, 2417230, 5256451], LEAF1), ([2162, 2286, 4027, 16494592, 8163328, 9363456, 38352715776, 31490834432, 33436991488], [2042, 745, 2546, 14716424, 8366074, 3052265], LEAF1), ([701, 510, 393, 1609728, 3358720, 2088960, 8556380160, 1996488704, 13757317120], [3123, 2106, 1891, 11864912, 12794931, 8628282], LEAF1), ([2069, 3943, 1947, 7974912, 16310272, 16150528, 66152562688, 38587596800, 66806874112], [1984, 1628, 2527, 14810655, 8128448, 6669916], LEAF1), ([160, 2184, 3056, 12517376, 11862016, 8945664, 36641439744, 14898167808, 48586817536], [3785, 3926, 1070, 11434727, 15507145, 16084822], LEAF1)]}

def cached_inverse(b, x, plan, with_zero=True):
    n = plan['n']
    h = n // 2
    to, frm, K = tower(n)
    linear_inplace(b, x, to)
    terms = plan['terms']
    with b.mark('inverse%d norm' % n): raw = raw_cache(b, x, plan)
    D = raw[:h]
    linear_xor(b, x, D, plan['affine'])
    with b.mark('inverse%d scalar' % n):
        value = inverse6_fixed18(b, D, with_zero) if h == 6 else cached_inverse(b, D, PLAN12, with_zero)
    k, F = value
    linear_xor(b, x, D, plan['affine'])
    linear_inplace(b, raw, plan['reduce'])
    kept = plan['size'] - len(plan['retire'])
    for q, row in zip(raw[kept:], plan['retire']):
        mask = b.alloc_bits(1)[0]
        b.hmr(q, mask)
        for j in bits(row):
            left, right, _ = terms[j]
            with ref.expose_pair(b, x, left << 1, right << 1) as (p, r): b.ccz(p, r, mask)
        b.free(mask)
        b.free(q)
    out = list(raw[:kept]) + list(b.alloc_qubits(n - kept))
    linear_inplace(b, out, plan['out'])
    source = list(x) + list(k)
    with b.mark('inverse%d products' % n):
        for forms, vectors, leaf in plan['leaves']:
            outputs, gates, low, offset = leaf

            def remap(form):
                result = form & 1
                for j in bits(form >> 1): result ^= forms[j] << 1
                return result
            for e, v in zip(outputs, vectors):
                if e & 1:
                    for j in bits(v): b.x(out[j])
                for a in bits((e & (1 << low) - 1) >> 1):
                    for i in bits(forms[a]):
                        for j in bits(v): b.cx(source[i], out[j])
            for i, (left, right) in enumerate(gates):
                vec = 0
                for e, v in zip(outputs, vectors):
                    if e >> offset + i & 1: vec ^= v
                with ref.expose_pair(b, source, remap(left), remap(right)) as (p, q): rankone_xor(b, p, q, out, vec)
    ar._erase_norm(b, out, k, K)
    b.free(k)
    linear_inplace(b, x, frm)
    linear_inplace(b, out, frm)
    return (out, F)

def inverse24(b, x):
    return cached_inverse(b, x, PLAN24, True)
inverse6_zero18 = inverse6_zero18_plan
TO6 = [1, 17, 37, 54, 49, 30]
FROM6 = [1, 14, 23, 58, 3, 18]

def raw_cache(b, x, plan):
    raw = b.alloc_qubits(plan['size'])
    for left, right, vec in plan['terms']:
        with ref.expose_pair(b, x, left << 1, right << 1) as (p, q): rankone_xor(b, p, q, raw, vec)
    return raw
W14_JOINT18 = {'K': 1, 'scale': 1, 'raw_gates': [[2, 16], [14, 112], [8, 64], [10, 80], [12, 96], [6, 48]], 'gates': [[63688, 65242], [47190, 50256], [42244, 64160], [42064, 65236], [37240, 62320], [11090, 43290], [9360, 33288], [48208, 51212], [8560, 51442]], 'outputs': [31410304, 13183220, 23895034, 16089926, 14693266, 6608146, 67091], 'AND': 18}

def phase_inverse6_heldflag17(b, x, L, F, zero_mask=None):
    """Apply inverse phase; consume the supplied exact F=[x=0]."""
    dag = W14_JOINT18
    frm = FROM6
    linear_inplace(b, x, TO6)
    g = b.alloc_qubits(6)
    for j, (l, r) in enumerate(dag['raw_gates']):
        with ref.expose_pair(b, x, l, r) as (p, q): b.init_and(p, q, g[j])
    prefix = list(x) + list(g)
    forms = ref.norm_forms(1)
    w = ref.inverse3_virtual(b, prefix, forms)
    source = prefix + list(w)
    trans = [sum(((frm[i] >> j & 1) << i for i in range(6))) for j in range(6)]
    linear_inplace(b, L, trans)
    for j, e in enumerate(dag['outputs'][:6]):
        uses_g0 = e >> 16 & 1
        lin = e & 65535 ^ (1555 if uses_g0 else 0)
        if lin & 1: b.z(L[j])
        for r in bits(lin >> 1): b.cz(L[j], source[r])
        if uses_g0: b.cz(F, L[j])
    for i, (l, r) in enumerate(dag['gates'][1:], 1):
        vector = sum(((e >> 16 + i & 1) << j for j, e in enumerate(dag['outputs'][:6])))
        js = list(bits(vector))
        if not js: continue
        pL = js[0]
        for j in js[1:]: b.cx(L[j], L[pL])
        with ref.expose_pair(b, source, l, r) as (p, q): b.ccz(p, q, L[pL])
        for j in reversed(js[1:]): b.cx(L[j], L[pL])
    if zero_mask is not None: b.cz(F, zero_mask)
    fm = b.alloc_bits(1)[0]
    b.hmr(F, fm)
    b.free(F)
    b.z(fm)
    for r in bits(1555 >> 1): b.cz(source[r], fm)
    with ref.expose_pair(b, source, *dag['gates'][0]) as (p, q): b.ccz(p, q, fm)
    b.free(fm)
    m = b.alloc_bits(3)
    b.hmr(w, m)
    b.free(w)
    ref.phase_inverse3_virtual(b, prefix, forms, m)
    b.free(m)
    for i, (l, r) in reversed(list(enumerate(dag['raw_gates']))):
        with ref.expose_pair(b, x, l, r) as (p, q): b.del_and(p, q, g[i])
    b.free(g)
    linear_inplace(b, L, inverse_columns(trans))
    linear_inplace(b, x, frm)
RAW = [[2, 16], [14, 112], [8, 64], [10, 80], [12, 96], [6, 48]]
PLAN = [('r', [397, 80, 58, 2, 153, 25, 9, 5, 1], 9, []), ('g', 1, None, None), ('p', 2948, 4256, 1), ('r', [1, 2, 4, 8, 16, 64, 304, 32, 128], 9, []), ('g', 1, None, None), ('p', 7928, 12528, 2), ('r', [1, 2, 4, 8, 16, 32, 160, 64, 256], 9, []), ('g', 1, None, None), ('p', 19728, 9864, 4), ('r', [1, 2, 4, 8, 16, 32, 64, 128, 256], 9, []), ('p', 17360, 46220, 7), ('r', [1, 2, 4, 8, 16, 32, 64, 256, 128], 8, [None]), ('p', 27760, 21746, 13), ('r', [1, 2, 4, 8, 16, 32, 64, 128], 7, [None]), ('p', 7250, 15002, 27), ('r', [1, 2, 4, 8, 16, 32, 64], 5, [(11, [(4, 2), (4, 3), (5, 1), (6, 1), (6, 3)]), None]), ('g', 1, None, None), ('p', 200, 474, 64), ('r', [17, 14, 3, 10, 12], 4, [(104, [(4, 1), (4, 2), (5, 1), (5, 3), (6, 2), (7, 0), (8, 0), (10, 0)])]), ('p', 214, 464, 41), ('r', [5, 10, 1, 2], 2, [(62, [(4, 2), (4, 3), (5, 1), (5, 3), (6, 1), (6, 2), (7, 0)]), (45, [(1, 0), (3, 0), (3, 2), (4, 0), (4, 1), (4, 2), (4, 3), (5, 0), (5, 3), (5, 4), (6, 0), (6, 1), (6, 2), (6, 5), (7, 0), (7, 2), (7, 3), (7, 6), (8, 0), (8, 1), (8, 2), (8, 3), (8, 7), (9, 5), (9, 6), (9, 7), (10, 0), (10, 1), (10, 4), (10, 5), (10, 7), (10, 8), (11, 0), (11, 2), (11, 3), (11, 5), (11, 9), (11, 10)])]), ('p', 208, 468, 59), ('r', [1, 2], 0, [(126, [(2, 0), (2, 1), (3, 0), (4, 1), (4, 2), (4, 3), (5, 0), (5, 4), (6, 0), (6, 1), (7, 2), (7, 4), (8, 0), (8, 7), (9, 2), (9, 4), (10, 0), (10, 1), (10, 6), (10, 7), (10, 9), (11, 7), (11, 8), (12, 8), (12, 9)]), (75, [(2, 0), (2, 1), (4, 2), (5, 1), (5, 2), (6, 2), (6, 3), (6, 4), (7, 0), (7, 2), (7, 3), (7, 4), (8, 2), (8, 5), (9, 1), (9, 6), (9, 8), (10, 2), (10, 8)])])]
OUTPUT = [45, 9, 63, 23, 15, 59, 75]
LINEAR = [0, 116, 122, 70, 18, 18, 19]

def inverse6_joint18_12extra(b, x):
    frm = FROM6
    linear_inplace(b, x, TO6)
    raw = b.alloc_qubits(6)
    for j, (l, r) in enumerate(RAW):
        with ref.expose_pair(b, x, l, r) as (p, q): b.init_and(p, q, raw[j])
    w = ref.inverse3_virtual(b, list(x) + list(raw), ref.norm_forms(1))
    raw = list(raw) + list(w)
    out = []
    for kind, a, c, d in PLAN:
        if kind == 'r':
            linear_inplace(b, raw, a)
            for j, action in enumerate(d, c):
                if action is None:
                    out.append(raw[j])
                    continue
                coeff, phase = action
                for k in bits(coeff): b.cx(raw[j], out[k])
                m = b.alloc_bits(1)[0]
                b.hmr(raw[j], m)
                live = list(x) + raw[:c] + raw[j + 1:] + out
                for i, k in phase:
                    if i == 0: b.z(m)
                    elif k == 0 or i == k: b.cz(live[i - 1], m)
                    else:
                        b.ccz(live[i - 1], live[k - 1], m)
                b.free(m)
                b.free(raw[j])
            raw = list(raw[:c])
        elif kind == 'g': out.extend(b.alloc_qubits(a))
        else:
            with ref.expose_pair(b, list(x) + raw, a, c) as (p, q): rankone_xor(b, p, q, out, d)
    linear_inplace(b, out, OUTPUT)
    for j, form in enumerate(LINEAR): ref.form_xor(b, x, form, out[j])
    linear_inplace(b, x, frm)
    linear_inplace(b, out[:6], frm)
    return (out[:6], out[6])

def inverse6_fixed18(b, x, joint=True):
    return inverse6_joint18_12extra(b, x)
