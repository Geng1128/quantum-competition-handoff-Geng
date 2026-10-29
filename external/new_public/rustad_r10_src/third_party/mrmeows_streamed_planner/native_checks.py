import numpy as np
import kickmix as km

def check(circuit, delta, *, brot=11, repeats=4, seed=127813):
    """Check every valid address with arbitrary targets, phase and scratch."""
    count, word_count = delta.shape
    n_sel = (count - 1).bit_length()
    addresses = np.tile(np.arange(count), repeats)
    rng = np.random.default_rng(seed)
    initial = rng.integers(0, 1 << brot, size=(len(addresses), word_count))
    if repeats >= 2:
        initial[:count] = 0
        initial[count:2*count] = (1 << brot) - 1
    sim = km.Simulator(batch_size=len(addresses), seed=seed, counting_operations=True)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots("selection", ((addresses[None, :] >> np.arange(n_sel)[:, None]) & 1).astype(bool))
    for j in range(word_count):
        sim.write_across_shots(f"word_{j}", ((initial[:, j][None, :] >> np.arange(brot)[:, None]) & 1).astype(bool))
    powers = np.int64(1) << np.arange(brot)
    for application in range(2):
        sim.do(circuit)
        got = np.stack([powers @ sim.read_across_shots(f"word_{j}").astype(np.int64)
                        for j in range(word_count)], axis=1)
        expected = initial ^ delta[addresses] if application == 0 else initial
        assert np.array_equal(got, expected), "incorrect arbitrary-target XOR"
        assert not np.any(sim.read_phase_flipped_across_shots()), "incorrect phase"
        decoded = (np.int64(1) << np.arange(n_sel)) @ sim.read_across_shots("selection").astype(np.int64)
        assert np.array_equal(decoded, addresses), "selection changed"
        total = sim.read_across_shots(km.array([km.q(i) for i in range(circuit.num_qubits)]))
        assert not np.any(total[n_sel + brot * word_count:]), "scratch not returned to zero"
    return float(sim.operation_statistics().num_toffoli_like) / len(addresses) / 2
