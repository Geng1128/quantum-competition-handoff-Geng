"""Our DQI decoder as decode(b, S, E), for check_decoder.py.

The stages, in order: mask projection, the initial linear maps, extraction (the jet kernel 289 times),
synthesis, conversion, scan, then conversion, synthesis, extraction and the linear maps undone, and the masks
restored. The jet kernel is exp/jet24/pro_jet.py: 683 Toffolis a pair (forward 290, reverse 393) at 43 private
qubits, which sets Q at 883. The undo of the synthesis uses two narrower divisions than the forward's library
default, so that it fits within 883 too: see unsynthesize below.
Written for the q2 kit by Halley (Claude Opus 5.5), 2026-10-27, from the decoder's own build script.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE), str(HERE / 'exp/decoder')]

from common import *  # noqa: E402,F401,F403  (km is circ; ROOT is this folder)
from exp.confluent_extract.kernel import initial, extract as extract_jets  # noqa: E402
from exp.confluent_extract.model import constants as confluent_constants  # noqa: E402
from exp.jet24.pro_jet import jet_forward as _jet_forward, jet_reverse as _jet_reverse  # noqa: E402
import exp.fold_cached_norm.kernel as FK  # noqa: E402
from exp.fold_joint.kernel import divide414_callback  # noqa: E402
from exp.mask_projection.kernel import project, restore, SELECTED  # noqa: E402
sys.path.insert(0, str(HERE / 'exp/convert'))
from exp.convert_short_hankel.phase34 import convert_safe as convert_historyfree  # noqa: E402
sys.path.insert(0, str(HERE / 'exp/scan'))
from scan import scan  # noqa: E402


def jet_forward(b, u, v, K=None):
    assert K in (None, 2048)
    return _jet_forward(b, u, v)


def jet_reverse(b, u, v, token, K=None):
    assert K in (None, 2048)
    return _jet_reverse(b, u, v, token)


def extract(b, words, reverse=False, transcript=None):
    return extract_jets(b, words, jet_forward, jet_reverse, reverse, transcript)


def divide_zero414(b, x, y, K=None, extra_zero_mask=None, after=None):
    """The outer fold's identity-at-zero division x /= y, at 25 private qubits: 414 Toffolis (the wide one, 324 at
    49, would set Q at 889 here, since 840 qubits are live)."""
    divide414_callback(b, x, y, K, extra_zero_mask, after)


synthesize = FK.synthesize
# The forward synthesis multiplies; only its undo divides. At the inner folds' divisions 853 qubits are live, so
# they use the 24-scratch division of exp/mul99/schedules.py (369 Toffolis) instead of the retained-norm one
# (323 at 36 private), and the outer fold uses divide_zero414.
unsynthesize = FK.bind(FK.prior.unsynthesize, absorb=FK.prior.absorb_odd, absorb_even=FK.prior.absorb_even,
                       outer=FK.bind(FK.prior.outer, divide_zero369=divide_zero414), odd_decode=FK.odd_decode,
                       multiply_zero144=FK.multiply_zero141)


def decode(b, S, E):
    with b.mark('mask_projection'):
        mask_token = project(b, S, E)
    with b.mark('initial'):
        initial(b, S)
    with b.mark('extract'):
        xh = extract(b, S)
    with b.mark('synthesis'):
        A, B, yh = synthesize(b, S)
    history_word = b.alloc_qubits(12)
    with b.mark('convert'):
        phase_token = convert_historyfree(b, A, B, history_scratch12=history_word, monic_root=True)
    with b.mark('scan'):
        scan(b, A, B, E, skip_points=SELECTED, node_polynomial=confluent_constants()[2])
    with b.mark('unconvert'):
        convert_historyfree(b, A, B, True, history_scratch12=history_word, phase_token=phase_token)
    b.free(history_word)
    with b.mark('unsynthesis'):
        unsynthesize(b, yh)
    with b.mark('unextract'):
        extract(b, S, True, xh)
    with b.mark('uninitial'):
        initial(b, S, True)
    with b.mark('restore_masks'):
        restore(b, E, mask_token)
