import math
import numpy as np


def _xlogx(x):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    mask = x > 0
    out[mask] = x[mask] * np.log(x[mask])
    return out


def _log1pexp(x):
    return np.log1p(np.exp(x))


def entropy(mean_vec):
    mean_vec = np.asarray(mean_vec, dtype=float)
    return -_xlogx(mean_vec) - _xlogx(1 - mean_vec)


def mi2(mean_vec, corr):
    """Pairwise mutual information contribution matrix (0/1 spins)."""
    mean_vec = np.asarray(mean_vec, dtype=float).reshape(-1, 1)
    corr = np.asarray(corr, dtype=float)
    a = -_xlogx(mean_vec) - _xlogx(1 - mean_vec)
    a = a + a.T
    b = _xlogx(corr)
    mc = mean_vec - corr
    c = _xlogx(mc) + _xlogx(mc).T
    d = _xlogx(1 - mean_vec - mean_vec.T + corr)
    delta_s = np.real(a + b + c + d)
    np.fill_diagonal(delta_s, 0.0)
    delta_s = np.nan_to_num(delta_s, nan=0.0, posinf=0.0, neginf=0.0)
    return delta_s


def energy_ising(spin, J, h):
    """Energy for a 0/1 Ising configuration."""
    spin = np.asarray(spin, dtype=float).reshape(-1, 1)
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float).reshape(-1, 1)
    energy = -0.5 * (spin.T @ J @ spin) - (h.T @ spin)
    return np.asarray(energy).item()


def _int_to_binary_vector(value, num_bits):
    bits = [(value >> i) & 1 for i in range(num_bits)]
    return np.array(bits, dtype=float)


def boltzman_ent(J, h, kT):
    """Exact Boltzmann entropy by full enumeration (0/1 spins)."""
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    num_spins = len(h)
    z = 0.0
    for i in range(2 ** num_spins):
        spin_set = _int_to_binary_vector(i, num_spins)
        z += math.exp(-energy_ising(spin_set, J, h) / kT)
    entropy_val = 0.0
    for i in range(2 ** num_spins):
        spin_set = _int_to_binary_vector(i, num_spins)
        prob = math.exp(-energy_ising(spin_set, J, h) / kT) / z
        if prob > 0:
            entropy_val -= prob * math.log(prob)
    return entropy_val


def exact_ising(J, h, kT):
    """Exact means/correlations/3-point and heat capacity by enumeration."""
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    num_spins = len(h)
    z = 0.0
    for i in range(2 ** num_spins):
        spin_set = _int_to_binary_vector(i, num_spins)
        z += math.exp(-energy_ising(spin_set, J, h) / kT)
    mean_spin = np.zeros(num_spins)
    corr = np.zeros((num_spins, num_spins))
    threepoint = np.zeros((num_spins, num_spins, num_spins))
    energy = 0.0
    c_val = 0.0
    for i in range(2 ** num_spins):
        spin_set = _int_to_binary_vector(i, num_spins)
        prob = math.exp(-energy_ising(spin_set, J, h) / kT) / z
        mean_spin += spin_set * prob
        corr += prob * np.outer(spin_set, spin_set)
        aa, bb, cc = np.meshgrid(spin_set, spin_set, spin_set, indexing="ij")
        tensor = aa * bb * cc
        threepoint += prob * tensor
        en = energy_ising(spin_set, J, h)
        energy += prob * en
        c_val += prob * (en ** 2)
    c_val = c_val - energy ** 2
    c_val = c_val / (kT ** 2)
    return mean_spin, corr, c_val, threepoint


def estimate_num_spike_dist(J, h, kT):
    """Estimate spike-count distribution via Glauber sampling (0/1 spins)."""
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    num_spins = len(h)
    spin_set = np.zeros(num_spins)
    spin_set[np.random.permutation(num_spins)[:3]] = 1
    num_iters = 2 ** 12
    probs = np.zeros(num_spins + 1)
    for _ in range(num_iters):
        for _ in range(num_spins):
            idx = np.random.randint(num_spins)
            dE = ((2 * spin_set[idx] - 1) * (J[idx, :] @ spin_set) - J[idx, idx]
                  + h[idx] * (2 * spin_set[idx] - 1))
            prob = math.exp(-dE / kT) / (1 + math.exp(-dE / kT))
            if np.random.rand() <= prob:
                spin_set[idx] = 1 - spin_set[idx]
        probs[int(spin_set.sum())] += 1
    probs = probs / probs.sum()
    spikes = np.arange(num_spins + 1)
    return spikes, probs


def glauber(J, h, kT):
    """Estimate means/correlations and specific heat via Glauber dynamics."""
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    num_spins = len(h)
    spin_set = np.zeros(num_spins)
    spin_set[np.random.permutation(num_spins)[: math.ceil(num_spins * 0.1)]] = 1
    num_iters = 2 ** 16
    means = np.zeros(num_spins)
    corrs = np.zeros((num_spins, num_spins))
    energy = 0.0
    energy_sq = 0.0
    for _ in range(num_iters):
        for _ in range(num_spins):
            idx = np.random.randint(num_spins)
            dE = ((2 * spin_set[idx] - 1) * (J[idx, :] @ spin_set) - J[idx, idx]
                  + h[idx] * (2 * spin_set[idx] - 1))
            prob = math.exp(-dE / kT) / (1 + math.exp(-dE / kT))
            if np.random.rand() <= prob:
                spin_set[idx] = 1 - spin_set[idx]
        means += spin_set
        corrs += np.outer(spin_set, spin_set)
        e_val = energy_ising(spin_set, J, h)
        energy += e_val
        energy_sq += e_val ** 2
    means /= num_iters
    corrs /= num_iters
    energy /= num_iters
    energy_sq /= num_iters
    specific_heat = (energy_sq - energy ** 2) / (kT ** 2)
    return means, corrs, specific_heat


def decimate_gsp(G0):
    """Randomly decimate degree-1/2 nodes in a GSP adjacency matrix.

    Returns:
      G: final adjacency after decimation
      D: (N-1) x 3 decimation order, with -1 for a missing neighbor
    """
    G0 = np.asarray(G0)
    n = G0.shape[0]
    spins = np.arange(n)
    G = (G0 != 0).astype(int)
    D = np.full((n - 1, 3), -1, dtype=int)
    degs = G.sum(axis=0)
    inds = np.where((degs <= 2) & (degs >= 1))[0]
    counter = 0
    while inds.size > 0:
        dtemp = np.full(3, -1, dtype=int)
        i = inds[np.random.randint(len(inds))]
        dtemp[0] = i
        Ni = spins[G[i, :].astype(bool)]
        dtemp[1] = int(Ni[0])
        G[i, Ni] = 0
        G[Ni, i] = 0
        prev_jk_con = 1
        if len(Ni) == 2:
            dtemp[2] = int(Ni[1])
            prev_jk_con = G[Ni[0], Ni[1]]
            G[Ni[0], Ni[1]] = 1
            G[Ni[1], Ni[0]] = 1
        degs[i] -= 2
        if prev_jk_con == 1:
            degs[Ni] -= 1
        inds = spins[(degs == 1) | (degs == 2)]
        D[counter, :] = dtemp
        counter += 1
    return G, D


def decimate_gsp_carefully(G0, h, keep):
    """Decimate a GSP graph while preserving nodes in keep."""
    G0 = np.asarray(G0, dtype=float)
    h = np.asarray(h, dtype=float)
    n = G0.shape[0]
    spins = np.arange(n)
    heff = h.copy()
    Jeff = G0.copy()
    G = (G0 != 0).astype(int)
    D = np.full((n - 1, 3), -1, dtype=int)
    degs = G.sum(axis=0)
    inds = np.where((degs <= 2) & (degs >= 1))[0]
    keep = np.asarray(keep, dtype=int)
    if keep.size > 0:
        inds = np.setdiff1d(inds, keep)
    counter = 0
    while inds.size > 0:
        dtemp = np.full(3, -1, dtype=int)
        i = inds[np.random.randint(len(inds))]
        dtemp[0] = i
        Ni = spins[G[i, :].astype(bool)]
        dtemp[1] = int(Ni[0])
        G[i, Ni] = 0
        G[Ni, i] = 0
        heff[Ni[0]] = heff[Ni[0]] - _log1pexp(heff[i]) + _log1pexp(heff[i] + Jeff[i, Ni[0]])
        prev_jk_con = 1
        if len(Ni) == 2:
            dtemp[2] = int(Ni[1])
            G[Ni[0], Ni[1]] = 1
            G[Ni[1], Ni[0]] = 1
            heff[Ni[1]] = heff[Ni[1]] - _log1pexp(heff[i]) + _log1pexp(heff[i] + Jeff[i, Ni[1]])
            Jeff[Ni[0], Ni[1]] = Jeff[Ni[0], Ni[1]] + _log1pexp(heff[i]) - _log1pexp(Jeff[i, Ni[0]] + heff[i]) - _log1pexp(Jeff[i, Ni[1]] + heff[i]) + _log1pexp(Jeff[i, Ni[0]] + Jeff[i, Ni[1]] + heff[i])
            Jeff[Ni[1], Ni[0]] = Jeff[Ni[0], Ni[1]]
        Jeff[i, Ni] = 0
        Jeff[Ni, i] = 0
        degs[i] -= 2
        if prev_jk_con == 1:
            degs[Ni] -= 1
        inds = spins[(degs == 1) | (degs == 2)]
        if keep.size > 0:
            inds = np.setdiff1d(inds, keep)
        D[counter, :] = dtemp
        counter += 1
    return G, D, Jeff, heff


def decimate(J, h):
    """Graph-based decimation with parameter renormalization (not used in core)."""
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    num_spins = len(h)
    Jeff = J.copy()
    heff = h.copy()
    parent1 = []
    parent2 = []
    child = []
    active = np.ones(num_spins, dtype=bool)
    while active.sum() > 1:
        active_idx = np.where(active)[0]
        degs = np.count_nonzero(Jeff[np.ix_(active_idx, active_idx)], axis=1)
        gind = active_idx[np.argmin(degs)]
        neigh = np.where(Jeff[gind, :] != 0)[0]
        child.append(gind)
        parent1.append(int(neigh[0]))
        heff[neigh[0]] = heff[neigh[0]] - _log1pexp(heff[gind]) + _log1pexp(heff[gind] + Jeff[gind, neigh[0]])
        if len(neigh) == 2:
            parent2.append(int(neigh[1]))
            heff[neigh[1]] = heff[neigh[1]] - _log1pexp(heff[gind]) + _log1pexp(heff[gind] + Jeff[gind, neigh[1]])
            if Jeff[neigh[0], neigh[1]] == 0:
                Jeff[neigh[0], neigh[1]] = _log1pexp(heff[gind]) - _log1pexp(Jeff[gind, neigh[0]] + heff[gind]) - _log1pexp(Jeff[gind, neigh[1]] + heff[gind]) + _log1pexp(Jeff[gind, neigh[0]] + Jeff[gind, neigh[1]] + heff[gind])
                Jeff[neigh[1], neigh[0]] = Jeff[neigh[0], neigh[1]]
            else:
                weight = Jeff[neigh[0], neigh[1]]
                Jeff[neigh[0], neigh[1]] = weight + _log1pexp(heff[gind]) - _log1pexp(Jeff[gind, neigh[0]] + heff[gind]) - _log1pexp(Jeff[gind, neigh[1]] + heff[gind]) + _log1pexp(Jeff[gind, neigh[0]] + Jeff[gind, neigh[1]] + heff[gind])
                Jeff[neigh[1], neigh[0]] = Jeff[neigh[0], neigh[1]]
        else:
            parent2.append(-1)
        active[gind] = False
    last_node = np.where(active)[0][0]
    child.append(int(last_node))
    parent1.append(-1)
    parent2.append(-1)
    return np.array(child, dtype=int), np.array(parent1, dtype=int), np.array(parent2, dtype=int), heff, Jeff


def jpairs(J, D):
    """Map decimation-order edges to a compact index set."""
    J = np.asarray(J)
    D = np.asarray(D, dtype=int)
    Js = np.full_like(J, -1, dtype=int)
    pairs = np.zeros((2*J.shape[0], 2), dtype=int)
    count = 0
    for i in range(D.shape[0]):
        a = D[i, 0]
        b = D[i, 1]
        c = D[i, 2]
        if a == -1 or b == -1:
            continue
        Js[a, b] = count
        pairs[count, 0] = a
        pairs[count, 1] = b
        count += 1
        if c != -1:
            Js[a, c] = count
            pairs[count, 0] = a
            pairs[count, 1] = c
            count += 1
    Js = np.where(Js != -1, Js, -1)
    Js = np.maximum(Js, Js.T)
    return Js, count, pairs[:count, :]


def find_max_in_selected(matrix, selected_rows, selected_cols):
    """Max of a submatrix, with indices mapped back to the original."""
    matrix = np.asarray(matrix)
    selected_rows = np.asarray(selected_rows, dtype=int)
    selected_cols = np.asarray(selected_cols, dtype=int)
    submatrix = matrix[np.ix_(selected_rows, selected_cols)]
    linear_idx = np.argmax(submatrix)
    row_idx, col_idx = np.unravel_index(linear_idx, submatrix.shape)
    true_row_idx = int(row_idx)
    true_col_idx = int(col_idx)
    row = int(selected_rows[row_idx])
    col = int(selected_cols[col_idx])
    max_val = submatrix[row_idx, col_idx]
    return max_val, row, col, true_row_idx, true_col_idx


def find_tree(mean_vec, corr):
    mean_vec = np.asarray(mean_vec, dtype=float)
    corr = np.asarray(corr, dtype=float)
    ent = float(np.sum(entropy(mean_vec)))
    num_spins = len(mean_vec)
    delta_s = mi2(mean_vec, corr)
    delta_s = delta_s - np.diag(np.diag(delta_s))
    spins = np.arange(num_spins)
    jguess = np.zeros((num_spins, num_spins))
    ents = np.zeros(num_spins)
    ents[0] = ent
    hguess = np.log(mean_vec / (1 - mean_vec))
    idx = np.unravel_index(np.argmax(delta_s), delta_s.shape)
    row, col = int(idx[0]), int(idx[1])
    delta_s[row, col] = 0
    delta_s[col, row] = 0
    hguess[row] = math.log((mean_vec[row] - corr[row, col]) / (1 + corr[row, col] - mean_vec[row] - mean_vec[col]))
    jguess[row, col] = math.log(corr[row, col] / (mean_vec[col] - corr[row, col])) - hguess[row]
    jguess[col, row] = jguess[row, col]
    hguess[col] = hguess[col] + _log1pexp(hguess[row]) - _log1pexp(jguess[row, col] + hguess[row])
    num_spins = num_spins - 2
    connected_spins = [row, col]
    spins = spins[(spins != row) & (spins != col)]
    ent = ent - delta_s[row, col]
    ents[1] = ent
    for i in range(num_spins):
        mval, row, col, trow, _ = find_max_in_selected(delta_s, spins, connected_spins)
        ent -= mval
        ents[i + 2] = ent
        connected_spins.append(row)
        spins = np.delete(spins, trow)
        delta_s[row, col] = 0
        delta_s[col, row] = 0
        hguess[row] = math.log((mean_vec[row] - corr[row, col]) / (1 + corr[row, col] - mean_vec[row] - mean_vec[col]))
        jguess[row, col] = math.log(corr[row, col] / (mean_vec[col] - corr[row, col])) - hguess[row]
        jguess[col, row] = jguess[row, col]
        hguess[col] = hguess[col] + _log1pexp(hguess[row]) - _log1pexp(jguess[row, col] + hguess[row])
    return jguess, hguess, ent, ents


def inverse_ising_gsp_01_helper(m, C, step_size):
    """Solve the 3-node inverse problem (0/1 spins).

    Inputs are magnetizations and correlations for nodes (1,2,3), where
    node 1 is the node being added back and nodes 2/3 are its neighbors.
    Returns h1, J12, J13 using a Newton update on the nonlinear equations.
    """
    threshold = 1e-10
    m1, m2, m3 = m
    c12 = C[0, 1]
    c13 = C[0, 2]
    c23 = C[1, 2]
    denom = (c23 ** 2 - 2 * c23 * m2 * m3 - m2 * m3 * (1 - m2 - m3))
    h1 = -2 * ((1 - 2 * m1) * c23 ** 2 - m2 * m3 * (1 + 2 * c12 + 2 * c13 - 2 * m1 - m2 - m3)
               + 2 * c23 * (c13 * m2 + (c12 - m2) * m3)) / denom
    j12 = 4 * (m1 * m3 * (m2 - c23) - c12 * m3 * (1 - m3) + c13 * (c23 - m2 * m3)) / denom
    j13 = 4 * (m1 * m2 * (m3 - c23) - c13 * m2 * (1 - m2) + c12 * (c23 - m2 * m3)) / denom
    diff = 1.0
    count = 1
    while diff > threshold:
        m1_temp = (1 - m2 - m3 + c23) / (1 + math.exp(-h1)) + (m2 - c23) / (1 + math.exp(-j12 - h1)) \
            + (m3 - c23) / (1 + math.exp(-j13 - h1)) + c23 / (1 + math.exp(-j12 - j13 - h1))
        c12_temp = (m2 - c23) / (1 + math.exp(-j12 - h1)) + c23 / (1 + math.exp(-j12 - j13 - h1))
        c13_temp = (m3 - c23) / (1 + math.exp(-j13 - h1)) + c23 / (1 + math.exp(-j12 - j13 - h1))
        dm1_dh1 = (1 - m2 - m3 + c23) * math.exp(-h1) / (1 + math.exp(-h1)) ** 2 \
            + (m2 - c23) * math.exp(-j12 - h1) / (1 + math.exp(-j12 - h1)) ** 2 \
            + (m3 - c23) * math.exp(-j13 - h1) / (1 + math.exp(-j13 - h1)) ** 2 \
            + c23 * math.exp(-j12 - j13 - h1) / (1 + math.exp(-j12 - j13 - h1)) ** 2
        dm1_dj12 = (m2 - c23) * math.exp(-j12 - h1) / (1 + math.exp(-j12 - h1)) ** 2 \
            + c23 * math.exp(-j12 - j13 - h1) / (1 + math.exp(-j12 - j13 - h1)) ** 2
        dm1_dj13 = (m3 - c23) * math.exp(-j13 - h1) / (1 + math.exp(-j13 - h1)) ** 2 \
            + c23 * math.exp(-j12 - j13 - h1) / (1 + math.exp(-j12 - j13 - h1)) ** 2
        dc12_dh1 = dm1_dj12
        dc12_dj12 = dm1_dj12
        dc12_dj13 = c23 * math.exp(-j12 - j13 - h1) / (1 + math.exp(-j12 - j13 - h1)) ** 2
        dc13_dh1 = dm1_dj13
        dc13_dj12 = dc12_dj13
        dc13_dj13 = dm1_dj13
        jac = np.array([[dm1_dh1, dm1_dj12, dm1_dj13],
                        [dc12_dh1, dc12_dj12, dc12_dj13],
                        [dc13_dh1, dc13_dj12, dc13_dj13]], dtype=float)
        rhs = np.array([m1_temp - m1, c12_temp - c12, c13_temp - c13], dtype=float)
        dhj = np.linalg.solve(jac, rhs)
        diff = float(np.sum(np.abs(dhj)))
        h1 -= step_size * dhj[0]
        j12 -= step_size * dhj[1]
        j13 -= step_size * dhj[2]
        if np.isnan([h1, j12, j13]).any() or max(abs(h1), abs(j12), abs(j13)) > 100:
            return h1, j12, j13
        count += 1
        if count > 1000:
            return h1, j12, j13
    return h1, j12, j13


def pointthree_mutual(J, h, mean_vec, corr, num_spins, spins, pair):
    """Compute candidate entropy drops for adding a third spin to a pair."""
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float).copy()
    mean_vec = np.asarray(mean_vec, dtype=float)
    corr = np.asarray(corr, dtype=float)
    delta_h = np.zeros(num_spins)
    j = pair[0]
    k = pair[1]
    for i in spins:
        if i == j or i == k:
            delta_h[i] = 0
            continue
        m = mean_vec[[i, j, k]]
        c = corr[np.ix_([i, j, k], [i, j, k])]
        step_size = 0.5
        h1, j12, j13 = inverse_ising_gsp_01_helper(m, c, step_size)
        h[i] = h1
        hjold = h[j]
        hkold = h[k]
        jkold = J[j, k]
        hj = h[j] + _log1pexp(h[i]) - _log1pexp(j12 + h[i])
        hk = h[k] + _log1pexp(h[i]) - _log1pexp(j13 + h[i])
        jk = J[j, k] - _log1pexp(h[i]) + _log1pexp(j12 + h[i]) + _log1pexp(j13 + h[i]) - _log1pexp(j12 + j13 + h[i])
        delta_h[i] = -_log1pexp(h[i]) + j12 * corr[i, j] + j13 * corr[i, k] + h[i] * mean_vec[i] \
            - mean_vec[i] * math.log(mean_vec[i]) - (1 - mean_vec[i]) * math.log(1 - mean_vec[i]) \
            + (jk - jkold) * corr[j, k] + (hj - hjold) * mean_vec[j] + (hk - hkold) * mean_vec[k]
    return delta_h


def find_gsp_update(mean_vec, corr):
    """Infer a GSP Ising model by greedy entropy reduction.

    Starts from independent spins, then adds interactions that maximize
    mutual-information/entropy drop at each step.
    """
    mean_vec = np.asarray(mean_vec, dtype=float)
    corr = np.asarray(corr, dtype=float)
    # Entropy of independent model
    ent = float(np.sum(entropy(mean_vec)))
    num_spins = len(mean_vec)
    ents = np.zeros(num_spins)
    ents[0] = ent
    # Initialize fields assuming independent spins
    h = np.log(mean_vec / (1 - mean_vec))
    J = np.zeros((num_spins, num_spins))
    binaryJ = np.zeros((num_spins, num_spins))
    # Pairwise mutual information matrix
    mis_temp = mi2(mean_vec, corr)
    idx = np.unravel_index(np.argmax(mis_temp), mis_temp.shape)
    start, ind = int(idx[0]), int(idx[1])
    ent -= mis_temp[start, ind]
    ents[1] = ent
    h[ind] = math.log((mean_vec[ind] - corr[ind, start]) / (1 + corr[ind, start] - mean_vec[start] - mean_vec[ind]))
    weight = math.log(corr[ind, start] / (mean_vec[start] - corr[ind, start])) - h[ind]
    h[start] = h[start] + _log1pexp(h[ind]) - _log1pexp(weight + h[ind])
    J[start, ind] = weight
    J[ind, start] = J[start, ind]
    binaryJ[start, ind] = 1
    binaryJ[ind, start] = 1
    unique_pairs_full = np.zeros((2 * num_spins - 3, 2), dtype=int)
    delta_hs = np.zeros((num_spins, 2 * num_spins - 3))
    pair = sorted([start, ind])
    connected_spins = pair.copy()
    spins = np.arange(num_spins)
    spins = np.setdiff1d(spins, connected_spins)
    delta_h = pointthree_mutual(J, h, mean_vec, corr, num_spins, spins, pair)
    delta_hs[:, 0] = delta_h
    unique_pairs_full[0, 0] = pair[0]
    unique_pairs_full[0, 1] = pair[1]
    count = 2
    # Main greedy loop: add one spin at a time
    while count < num_spins:
        linear_idx = np.argmax(delta_hs)
        i, col = np.unravel_index(linear_idx, delta_hs.shape)
        jk = unique_pairs_full[col, :]
        j = int(jk[0])
        k = int(jk[1])
        # Solve inverse problem for 3-spin subsystem (i,j,k)
        m = mean_vec[[i, j, k]]
        c = corr[np.ix_([i, j, k], [i, j, k])]
        step_size = 1.0
        h1, j12, j13 = inverse_ising_gsp_01_helper(m, c, step_size)
        h[i] = h1
        Jij = j12
        Jik = j13
        J[i, j] = Jij
        J[j, i] = Jij
        J[i, k] = Jik
        J[k, i] = Jik
        binaryJ[i, j] = 1
        binaryJ[j, i] = 1
        binaryJ[i, k] = 1
        binaryJ[k, i] = 1
        hjold = h[j]
        hkold = h[k]
        hj = h[j] + _log1pexp(h[i]) - _log1pexp(Jij + h[i])
        hk = h[k] + _log1pexp(h[i]) - _log1pexp(Jik + h[i])
        jk_val = J[j, k] - _log1pexp(h[i]) + _log1pexp(Jij + h[i]) + _log1pexp(Jik + h[i]) - _log1pexp(Jij + Jik + h[i])
        h[j] = hj
        h[k] = hk
        J[j, k] = jk_val
        J[k, j] = J[j, k]
        ent -= delta_hs[i, col]
        ents[count] = ent
        connected_spins = connected_spins + [int(i)]
        spins = np.setdiff1d(np.arange(num_spins), connected_spins)
        base_col = 2 * count - 3
        pair = sorted([i, j])
        delta_h = pointthree_mutual(J, h, mean_vec, corr, num_spins, spins, pair)
        delta_hs[:, base_col] = delta_h
        unique_pairs_full[base_col, 0] = pair[0]
        unique_pairs_full[base_col, 1] = pair[1]
        pair = sorted([i, k])
        delta_h = pointthree_mutual(J, h, mean_vec, corr, num_spins, spins, pair)
        delta_hs[:, base_col + 1] = delta_h
        unique_pairs_full[base_col + 1, 0] = pair[0]
        unique_pairs_full[base_col + 1, 1] = pair[1]
        delta_hs[i, :] = 0
        count += 1
    return h, J, ent, ents


def gsp_fit(datamean, datacorr, topology):
    """Fit a GSP network onto data given a topology."""
    topology = np.asarray(topology, dtype=float)
    num_spins = len(datamean)
    _, D = decimate_gsp(topology)
    D = D[: num_spins - 1, :]
    D = np.flipud(D)
    J = np.zeros((num_spins, num_spins))
    h = np.log(datamean / (1 - datamean))
    row = D[0, 0]
    col = D[0, 1]
    if row == col:
        col = col + 1
    h[row] = math.log((datamean[row] - datacorr[row, col]) / (1 + datacorr[row, col] - datamean[row] - datamean[col]))
    J[row, col] = math.log(datacorr[row, col] / (datamean[col] - datacorr[row, col])) - h[row]
    J[col, row] = J[row, col]
    h[col] = h[col] + _log1pexp(h[row]) - _log1pexp(J[row, col] + h[row])
    hind = float(np.sum(entropy(datamean)))
    h2 = mi2(datamean[[row, col]], datacorr[np.ix_([row, col], [row, col])])
    H = hind - h2[0, 1]
    for step in range(1, num_spins - 1):
        i = D[step, 0]
        j = D[step, 1]
        k = D[step, 2]
        m = datamean[[i, j, k]]
        c = datacorr[np.ix_([i, j, k], [i, j, k])]
        h1, j12, j13 = inverse_ising_gsp_01_helper(m, c, 1)
        h[i] = h1
        Jij = j12
        Jik = j13
        J[i, j] = Jij
        J[j, i] = Jij
        J[i, k] = Jik
        J[k, i] = Jik
        hjold = h[j]
        hkold = h[k]
        jkold = J[j, k]
        hj = h[j] + _log1pexp(h[i]) - _log1pexp(Jij + h[i])
        hk = h[k] + _log1pexp(h[i]) - _log1pexp(Jik + h[i])
        jk_val = J[j, k] - _log1pexp(h[i]) + _log1pexp(Jij + h[i]) + _log1pexp(Jik + h[i]) - _log1pexp(Jij + Jik + h[i])
        h[j] = hj
        h[k] = hk
        J[j, k] = jk_val
        J[k, j] = J[j, k]
        delta_h = -_log1pexp(h[i]) + J[i, j] * datacorr[i, j] + J[i, k] * datacorr[i, k] + h[i] * datamean[i] \
            - datamean[i] * math.log(datamean[i]) - (1 - datamean[i]) * math.log(1 - datamean[i]) \
            + (J[j, k] - jkold) * datacorr[j, k] + (h[j] - hjold) * datamean[j] + (h[k] - hkold) * datamean[k]
        H = H - delta_h
    return J, h, H


def gsp_fit_topology(mean_vec, corr, topo):
    """Fit a GSP network given a fixed topology."""
    num_spins = len(mean_vec)
    _, D = decimate_gsp(topo)
    D = np.flipud(D)
    J = np.zeros((num_spins, num_spins))
    h = np.log(mean_vec / (1 - mean_vec))
    row = D[0, 0]
    col = D[0, 1]
    h[row] = math.log((mean_vec[row] - corr[row, col]) / (1 + corr[row, col] - mean_vec[row] - mean_vec[col]))
    J[row, col] = math.log(corr[row, col] / (mean_vec[col] - corr[row, col])) - h[row]
    J[col, row] = J[row, col]
    h[col] = h[col] + _log1pexp(h[row]) - _log1pexp(J[row, col] + h[row])
    hind = float(np.sum(entropy(mean_vec)))
    h2 = mi2(mean_vec[[row, col]], corr[np.ix_([row, col], [row, col])])
    H = hind - h2[0, 1]
    for step in range(1, num_spins - 1):
        k = D[step, 2]
        j = D[step, 1]
        i = D[step, 0]
        m = mean_vec[[i, j, k]]
        c = corr[np.ix_([i, j, k], [i, j, k])]
        h1, j12, j13 = inverse_ising_gsp_01_helper(m, c, 1)
        h[i] = h1
        Jij = j12
        Jik = j13
        J[i, j] = Jij
        J[j, i] = Jij
        J[i, k] = Jik
        J[k, i] = Jik
        hjold = h[j]
        hkold = h[k]
        jkold = J[j, k]
        hj = h[j] + _log1pexp(h[i]) - _log1pexp(Jij + h[i])
        hk = h[k] + _log1pexp(h[i]) - _log1pexp(Jik + h[i])
        jk_val = J[j, k] - _log1pexp(h[i]) + _log1pexp(Jij + h[i]) + _log1pexp(Jik + h[i]) - _log1pexp(Jij + Jik + h[i])
        h[j] = hj
        h[k] = hk
        J[j, k] = jk_val
        J[k, j] = J[j, k]
        delta_h = -_log1pexp(h[i]) + Jij * corr[i, j] + Jik * corr[i, k] + h[i] * mean_vec[i] \
            - mean_vec[i] * math.log(mean_vec[i]) - (1 - mean_vec[i]) * math.log(1 - mean_vec[i]) \
            + (jk_val - jkold) * corr[j, k] + (hj - hjold) * mean_vec[j] + (hk - hkold) * mean_vec[k]
        H = H - delta_h
    return J, h, H


def random_gsp_fit(datamean, datacorr):
    """Fit a random GSP topology to data (random edge growth)."""
    num_spins = len(datamean)
    J = np.zeros((num_spins, num_spins))
    h = np.log(datamean / (1 - datamean))
    linidx = np.random.randint((num_spins) * (num_spins - 1) // 2)
    p = (math.sqrt(1 + 8 * linidx) - 1) / 2
    idx0 = math.floor(p)
    row = idx0 + 1
    col = linidx - idx0 * (idx0 + 1) / 2 + 1
    row = int(row)
    col = int(col)
    if row == col:
        col = col + 1
    connected_spins = [row, col]
    h[row] = math.log((datamean[row] - datacorr[row, col]) / (1 + datacorr[row, col] - datamean[row] - datamean[col]))
    J[row, col] = math.log(datacorr[row, col] / (datamean[col] - datacorr[row, col])) - h[row]
    J[col, row] = J[row, col]
    h[col] = h[col] + _log1pexp(h[row]) - _log1pexp(J[row, col] + h[row])
    spins = np.arange(num_spins)
    spins = np.setdiff1d(spins, connected_spins)
    hind = float(np.sum(entropy(datamean)))
    h2 = mi2(datamean[[row, col]], datacorr[np.ix_([row, col], [row, col])])
    H = hind - h2[0, 1]
    ents = np.zeros(num_spins)
    ents[0] = hind
    ents[1] = H
    addedpairs = np.zeros((2 * num_spins - 3, 2), dtype=int)
    addedpairs[0, 0] = row
    addedpairs[0, 1] = col
    for step in range(2, num_spins):
        randidx = np.random.randint(2 * step - 3)
        i = spins[np.random.randint(num_spins - step)]
        j = addedpairs[randidx, 0]
        k = addedpairs[randidx, 1]
        addedpairs[2 * step - 2, 0] = i
        addedpairs[2 * step - 2, 1] = j
        addedpairs[2 * step - 1, 0] = i
        addedpairs[2 * step - 1, 1] = k
        connected_spins.append(int(i))
        m = datamean[[i, j, k]]
        c = datacorr[np.ix_([i, j, k], [i, j, k])]
        h1, j12, j13 = inverse_ising_gsp_01_helper(m, c, 1)
        h[i] = h1
        Jij = j12
        Jik = j13
        J[i, j] = Jij
        J[j, i] = Jij
        J[i, k] = Jik
        J[k, i] = Jik
        hjold = h[j]
        hkold = h[k]
        jkold = J[j, k]
        hj = h[j] + _log1pexp(h[i]) - _log1pexp(Jij + h[i])
        hk = h[k] + _log1pexp(h[i]) - _log1pexp(Jik + h[i])
        jk_val = J[j, k] - _log1pexp(h[i]) + _log1pexp(Jij + h[i]) + _log1pexp(Jik + h[i]) - _log1pexp(Jij + Jik + h[i])
        h[j] = hj
        h[k] = hk
        J[j, k] = jk_val
        J[k, j] = J[j, k]
        delta_h = -_log1pexp(h[i]) + J[i, j] * datacorr[i, j] + J[i, k] * datacorr[i, k] + h[i] * datamean[i] \
            - datamean[i] * math.log(datamean[i]) - (1 - datamean[i]) * math.log(1 - datamean[i]) \
            + (J[j, k] - jkold) * datacorr[j, k] + (h[j] - hjold) * datamean[j] + (h[k] - hkold) * datamean[k]
        spins = np.setdiff1d(np.arange(num_spins), connected_spins)
        H = H - delta_h
        ents[step] = H
    return J, h, H, ents


def random_tree(datamean, datacorr):
    """Fit a random tree topology to data."""
    num_spins = len(datamean)
    spins = np.arange(num_spins)
    J_tree = np.zeros((num_spins, num_spins))
    h_tree = np.log(datamean / (1 - datamean))
    linidx = np.random.randint((num_spins) * (num_spins - 1) // 2)
    p = (math.sqrt(1 + 8 * linidx) - 1) / 2
    idx0 = math.floor(p)
    row = int(idx0 + 1)
    col = int(linidx - idx0 * (idx0 + 1) / 2 + 1)
    if row == col:
        col = col + 1
    h_tree[row] = math.log((datamean[row] - datacorr[row, col]) / (1 + datacorr[row, col] - datamean[row] - datamean[col]))
    J_tree[row, col] = math.log(datacorr[row, col] / (datamean[col] - datacorr[row, col])) - h_tree[row]
    J_tree[col, row] = J_tree[row, col]
    h_tree[col] = h_tree[col] + _log1pexp(h_tree[row]) - _log1pexp(J_tree[row, col] + h_tree[row])
    connected_spins = [row, col]
    spins = np.setdiff1d(spins, connected_spins)
    hind = float(np.sum(entropy(datamean)))
    h2 = mi2(datamean, datacorr)
    H = hind - h2[row, col]
    ents = np.zeros(num_spins)
    ents[0] = hind
    ents[1] = H
    for i in range(1, num_spins - 1):
        row = spins[np.random.randint(num_spins - i - 1)]
        col = connected_spins[np.random.randint(i + 1)]
        connected_spins.append(int(row))
        spins = np.setdiff1d(np.arange(num_spins), connected_spins)
        h_tree[row] = math.log((datamean[row] - datacorr[row, col]) / (1 + datacorr[row, col] - datamean[row] - datamean[col]))
        J_tree[row, col] = math.log(datacorr[row, col] / (datamean[col] - datacorr[row, col])) - h_tree[row]
        J_tree[col, row] = J_tree[row, col]
        h_tree[col] = h_tree[col] + _log1pexp(h_tree[row]) - _log1pexp(J_tree[row, col] + h_tree[row])
        H = H - h2[row, col]
        ents[i + 1] = H
    return J_tree, h_tree, H, ents


def random_gsp_ising(num_spins):
    """Generate a random GSP Ising model (J, h)."""
    J = np.zeros((num_spins, num_spins))
    h = np.random.randn(num_spins)
    J[0, 1] = np.random.randn()
    Jcons = J != 0
    p1 = [0]
    p2 = [1]
    for i in range(2, num_spins):
        randidx = np.random.randint(len(p1))
        if np.random.randint(8) != -2:
            J[p1[randidx], i] = np.random.randn()
            J[p2[randidx], i] = np.random.randn()
            Jcons[p1[randidx], i] = 1
            Jcons[p2[randidx], i] = 1
            p2 = [p1[randidx], p2[randidx]] + p2
            p1 = [i, i] + p1
        else:
            if np.random.randint(2) == 1:
                J[p1[randidx], i] = np.random.randn()
                Jcons[p1[randidx], i] = 1
                Jcons[p2[randidx], i] = 1
                p2 = [p1[randidx], p2[randidx]] + p2
                p1 = [i, i] + p1
            else:
                J[p2[randidx], i] = np.random.randn()
                Jcons[p1[randidx], i] = 1
                Jcons[p2[randidx], i] = 1
                p2 = [p1[randidx], p2[randidx]] + p2
                p1 = [i, i] + p1
    J = J + J.T
    return J, h


def minimum_distance_tree(datamean, datacorr, locations):
    """Fit a tree using inverse-distance heuristic."""
    locations = np.asarray(locations, dtype=float)
    diff = locations[:, None, :] - locations[None, :, :]
    distance_metric = np.sqrt((diff ** 2).sum(axis=-1))
    num_spins = distance_metric.shape[0]
    spins = np.arange(num_spins)
    distance_metric = distance_metric - np.diag(np.inf * np.ones(num_spins))
    J_tree = np.zeros((num_spins, num_spins))
    h_tree = np.log(datamean / (1 - datamean))
    delta_s = 1.0 / distance_metric
    delta_s = delta_s - np.diag(np.diag(delta_s))
    idx = np.unravel_index(np.argmax(delta_s), delta_s.shape)
    row, col = int(idx[0]), int(idx[1])
    delta_s[row, col] = 0
    delta_s[col, row] = 0
    h_tree[row] = math.log((datamean[row] - datacorr[row, col]) / (1 + datacorr[row, col] - datamean[row] - datamean[col]))
    J_tree[row, col] = math.log(datacorr[row, col] / (datamean[col] - datacorr[row, col])) - h_tree[row]
    J_tree[col, row] = J_tree[row, col]
    h_tree[col] = h_tree[col] + _log1pexp(h_tree[row]) - _log1pexp(J_tree[row, col] + h_tree[row])
    num_spins = num_spins - 2
    connected_spins = [row, col]
    spins = spins[(spins != row) & (spins != col)]
    hind = float(np.sum(entropy(datamean)))
    h2 = mi2(datamean, datacorr)
    H = hind - h2[row, col]
    for _ in range(num_spins):
        mval, row, col, trow, _ = find_max_in_selected(delta_s, spins, connected_spins)
        connected_spins.append(int(row))
        spins = np.delete(spins, trow)
        delta_s[row, col] = 0
        delta_s[col, row] = 0
        h_tree[row] = math.log((datamean[row] - datacorr[row, col]) / (1 + datacorr[row, col] - datamean[row] - datamean[col]))
        J_tree[row, col] = math.log(datacorr[row, col] / (datamean[col] - datacorr[row, col])) - h_tree[row]
        J_tree[col, row] = J_tree[row, col]
        h_tree[col] = h_tree[col] + _log1pexp(h_tree[row]) - _log1pexp(J_tree[row, col] + h_tree[row])
        H = H - h2[row, col]
    return J_tree, h_tree, H


def minimum_distance_gsp(datamean, datacorr, locations):
    """Fit a GSP using inverse-distance heuristic."""
    locations = np.asarray(locations, dtype=float)
    diff = locations[:, None, :] - locations[None, :, :]
    distance_metric = np.sqrt((diff ** 2).sum(axis=-1))
    num_spins = distance_metric.shape[0]
    spins = np.arange(num_spins)
    distance_metric = distance_metric + np.diag(np.inf * np.ones(num_spins))
    J_gsp = np.zeros((num_spins, num_spins))
    h_gsp = np.log(datamean / (1 - datamean))
    delta_s = 1.0 / distance_metric
    delta_s = delta_s - np.diag(np.diag(delta_s))
    idx = np.unravel_index(np.argmax(delta_s), delta_s.shape)
    row, col = int(idx[0]), int(idx[1])
    delta_s[row, col] = 0
    delta_s[col, row] = 0
    h_gsp[row] = math.log((datamean[row] - datacorr[row, col]) / (1 + datacorr[row, col] - datamean[row] - datamean[col]))
    J_gsp[row, col] = math.log(datacorr[row, col] / (datamean[col] - datacorr[row, col])) - h_gsp[row]
    J_gsp[col, row] = J_gsp[row, col]
    h_gsp[col] = h_gsp[col] + _log1pexp(h_gsp[row]) - _log1pexp(J_gsp[row, col] + h_gsp[row])
    connected_spins = [row, col]
    hind = float(np.sum(entropy(datamean)))
    h2 = mi2(datamean[[row, col]], datacorr[np.ix_([row, col], [row, col])])
    H = hind - h2[0, 1]
    addedpairs = np.zeros((2 * num_spins - 3, 2), dtype=int)
    addedpairs[0, 0] = col
    addedpairs[0, 1] = row
    distance_tri = np.zeros((num_spins, 2 * num_spins - 3))
    spins = np.setdiff1d(spins, connected_spins)
    for spin in spins:
        distance_tri[spin, 0] = 1.0 / (distance_metric[spin, row] + distance_metric[spin, col])
    for step in range(num_spins - 2):
        linear_idx = np.argmax(distance_tri)
        i, col_idx = np.unravel_index(linear_idx, distance_tri.shape)
        jk = addedpairs[col_idx, :]
        j = int(jk[0])
        k = int(jk[1])
        connected_spins.append(int(i))
        m = datamean[[i, j, k]]
        c = datacorr[np.ix_([i, j, k], [i, j, k])]
        h1, j12, j13 = inverse_ising_gsp_01_helper(m, c, 1)
        h_gsp[i] = h1
        Jij = j12
        Jik = j13
        J_gsp[i, j] = Jij
        J_gsp[j, i] = Jij
        J_gsp[i, k] = Jik
        J_gsp[k, i] = Jik
        hjold = h_gsp[j]
        hkold = h_gsp[k]
        jkold = J_gsp[j, k]
        hj = h_gsp[j] + _log1pexp(h_gsp[i]) - _log1pexp(Jij + h_gsp[i])
        hk = h_gsp[k] + _log1pexp(h_gsp[i]) - _log1pexp(Jik + h_gsp[i])
        jk_val = J_gsp[j, k] - _log1pexp(h_gsp[i]) + _log1pexp(Jij + h_gsp[i]) + _log1pexp(Jik + h_gsp[i]) - _log1pexp(Jij + Jik + h_gsp[i])
        h_gsp[j] = hj
        h_gsp[k] = hk
        J_gsp[j, k] = jk_val
        J_gsp[k, j] = J_gsp[j, k]
        delta_h = -_log1pexp(h_gsp[i]) + J_gsp[i, j] * datacorr[i, j] + J_gsp[i, k] * datacorr[i, k] + h_gsp[i] * datamean[i] \
            - datamean[i] * math.log(datamean[i]) - (1 - datamean[i]) * math.log(1 - datamean[i]) \
            + (J_gsp[j, k] - jkold) * datacorr[j, k] + (h_gsp[j] - hjold) * datamean[j] + (h_gsp[k] - hkold) * datamean[k]
        pair = sorted([i, j])
        addedpairs[2 * step, 0] = pair[0]
        addedpairs[2 * step, 1] = pair[1]
        pair = sorted([i, k])
        addedpairs[2 * step + 1, 0] = pair[0]
        addedpairs[2 * step + 1, 1] = pair[1]
        spins = np.setdiff1d(np.arange(num_spins), connected_spins)
        for spin in spins:
            distance_tri[spin, 2 * step] = 1.0 / (distance_metric[spin, i] + distance_metric[spin, j])
            distance_tri[spin, 2 * step + 1] = 1.0 / (distance_metric[spin, i] + distance_metric[spin, k])
        distance_tri[i, :] = 0
        H = H - delta_h
    return J_gsp, h_gsp, H


def overlap(respJ, spontJ):
    """Compute overlap fractions between two networks across thresholds."""
    respJ = np.abs(respJ)
    spontJ = np.abs(spontJ)
    neuron_num_spont = respJ.shape[0]
    trials = 2 * neuron_num_spont - 3
    overlapfrac = np.zeros(trials)
    overlapfracresp = np.zeros(trials)
    overlapfracspont = np.zeros(trials)
    sort_resp = np.sort(respJ[np.triu(respJ) != 0])
    sort_spont = np.sort(spontJ[np.triu(spontJ) != 0])
    sort_resp = np.concatenate(([0], sort_resp))
    sort_spont = np.concatenate(([0], sort_spont))
    for count in range(trials):
        brespJthres = respJ > sort_resp[count]
        bspontJthres = spontJ > sort_spont[count]
        samecons = (brespJthres + bspontJthres) > 1
        overlapfrac[count] = samecons.sum() / (2 * (2 * neuron_num_spont - 3 - count))
        samecons = (brespJthres + (spontJ != 0)) > 1
        overlapfracresp[count] = samecons.sum() / (2 * (2 * neuron_num_spont - 3 - count))
        samecons = ((respJ != 0) + bspontJthres) > 1
        overlapfracspont[count] = samecons.sum() / (2 * (2 * neuron_num_spont - 3 - count))
    fracs = (2 * neuron_num_spont - 3 - np.arange(trials) + 0) / (2 * neuron_num_spont - 3)
    return overlapfrac, overlapfracresp, overlapfracspont, fracs


def plotvariance(xdata, ydata, num_bins, color, ax=None):
    """Plot binned mean ± std envelope for scatter data."""
    import matplotlib.pyplot as plt
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    bin_edges = np.linspace(np.nanmin(xdata), np.nanmax(xdata), num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.digitize(xdata, bin_edges) - 1
    bin_means = np.zeros(num_bins)
    bin_stds = np.zeros(num_bins)
    for i in range(num_bins):
        data_in_bin = ydata[bin_idx == i]
        if data_in_bin.size == 0:
            bin_means[i] = np.nan
            bin_stds[i] = np.nan
        else:
            bin_means[i] = np.nanmean(data_in_bin)
            bin_stds[i] = np.nanstd(data_in_bin)
    curve1 = bin_means + bin_stds
    curve2 = bin_means - bin_stds
    if ax is None:
        ax = plt.gca()
    x2 = np.concatenate([bin_centers, bin_centers[::-1]])
    in_between = np.concatenate([curve1, curve2[::-1]])
    ax.fill(x2, in_between, color=color, alpha=0.1, linewidth=0)
    ax.plot(bin_centers, bin_means, color=color, linewidth=2)
    return ax


def _find_leaves(mi):
    mi = np.asarray(mi)
    deg = np.count_nonzero(mi, axis=1)
    leaves = np.where(deg == 1)[0]
    nodes = np.where(deg == 2)[0]
    if nodes.size == 0:
        node = np.array([], dtype=int)
    else:
        node = nodes
    return leaves, node


def chow_liu(i, j, k, mean_vec, corr):
    """Approximate a triangle by its maximum mutual-info subtree."""
    spins = [i, j, k]
    mi = mi2(mean_vec[spins], corr[np.ix_(spins, spins)])
    mi = np.triu(mi)
    mi = mi + np.tril(np.ones((3, 3)))
    row, col = np.unravel_index(np.argmin(mi), mi.shape)
    mi[row, col] = 0
    mi = np.triu(mi) + np.triu(mi).T - 2 * np.diag(np.diag(np.ones((3, 3))))
    leaves, node = _find_leaves(mi)
    unique_vals, counts = np.unique(np.concatenate([leaves, node]), return_counts=True)
    if len(unique_vals) == 3:
        for idx, val in enumerate(unique_vals):
            if counts[idx] == 2:
                first = spins[(val - 2) % 3]
                middle = spins[val]
                last = spins[val % 3]
        prob1 = corr[first, middle] * corr[middle, last] / mean_vec[middle]
    elif len(unique_vals) == 2:
        first = spins[unique_vals[0]]
        last = spins[unique_vals[1]]
        if 0 not in unique_vals:
            prob1 = corr[first, last] * mean_vec[spins[0]]
        if 1 not in unique_vals:
            prob1 = corr[first, last] * mean_vec[spins[1]]
        if 2 not in unique_vals:
            prob1 = corr[first, last] * mean_vec[spins[2]]
    else:
        prob1 = mean_vec[spins[0]] * mean_vec[spins[1]] * mean_vec[spins[2]]
    threepoint = np.zeros((3, 3, 3))
    threepoint[0, 1, 2] = prob1
    threepoint[0, 2, 1] = prob1
    threepoint[1, 0, 2] = prob1
    threepoint[1, 2, 0] = prob1
    threepoint[2, 0, 1] = prob1
    threepoint[2, 1, 0] = prob1
    threepoint[0, 0, 0] = mean_vec[spins[0]]
    threepoint[1, 1, 1] = mean_vec[spins[1]]
    threepoint[2, 2, 2] = mean_vec[spins[2]]
    threepoint[0, 1, 1] = corr[spins[0], spins[1]]
    threepoint[1, 0, 1] = threepoint[0, 1, 1]
    threepoint[1, 1, 0] = threepoint[0, 1, 1]
    threepoint[1, 0, 0] = threepoint[0, 1, 1]
    threepoint[0, 0, 1] = threepoint[0, 1, 1]
    threepoint[0, 1, 0] = threepoint[0, 1, 1]
    threepoint[0, 2, 2] = corr[spins[0], spins[2]]
    threepoint[2, 0, 2] = threepoint[0, 2, 2]
    threepoint[2, 2, 0] = threepoint[0, 2, 2]
    threepoint[2, 0, 0] = threepoint[0, 2, 2]
    threepoint[0, 0, 2] = threepoint[0, 2, 2]
    threepoint[0, 2, 0] = threepoint[0, 2, 2]
    threepoint[2, 1, 1] = corr[spins[1], spins[2]]
    threepoint[1, 2, 1] = threepoint[2, 1, 1]
    threepoint[1, 1, 2] = threepoint[2, 1, 1]
    threepoint[1, 2, 2] = threepoint[2, 1, 1]
    threepoint[2, 2, 1] = threepoint[2, 1, 1]
    threepoint[2, 1, 2] = threepoint[2, 1, 1]
    return threepoint


def entropy_gsp(J, h, exact_mean, exact_corr):
    """Compute GSP entropy using decimation and MI corrections."""
    J_eff = np.asarray(J, dtype=float).copy()
    h_eff = np.asarray(h, dtype=float).copy()
    _, D = decimate_gsp(J)
    H = float(np.sum(entropy(exact_mean)))
    H2 = mi2(exact_mean, exact_corr)
    c_val = 0.0
    for step in range(D.shape[0]):
        i = D[step, 0]
        j = D[step, 1]
        k = D[step, 2]
        if i == -1 or j == -1:
            continue
        h_eff[j] = h_eff[j] - _log1pexp(h_eff[i]) + _log1pexp(J_eff[j, i] + h_eff[i])
        if k != -1:
            h_eff[k] = h_eff[k] - _log1pexp(h_eff[i]) + _log1pexp(J_eff[k, i] + h_eff[i])
            J_eff[j, k] = J_eff[j, k] + _log1pexp(h_eff[i]) - _log1pexp(J_eff[i, j] + h_eff[i]) - _log1pexp(J_eff[i, k] + h_eff[i]) + _log1pexp(J_eff[i, j] + J_eff[i, k] + h_eff[i])
            J_eff[k, j] = J_eff[j, k]
    D = np.flipud(D)
    for i in range(len(h)):
        c_val += _log1pexp(h_eff[i])
    for i in range(D.shape[0]):
        xi = D[i, 0]
        xj = D[i, 1]
        xk = D[i, 2]
        if xi == -1 or xj == -1:
            continue
        if xk == -1:
            H = H - H2[xi, xj]
        else:
            mi3 = -_log1pexp(h_eff[xi]) + J_eff[xi, xj] * exact_corr[xi, xj] + J_eff[xi, xk] * exact_corr[xi, xk] \
                + h_eff[xi] * exact_mean[xi] - exact_mean[xi] * math.log(exact_mean[xi]) - (1 - exact_mean[xi]) * math.log(1 - exact_mean[xi]) \
                + (-_log1pexp(h_eff[xi]) + _log1pexp(J_eff[xi, xj] + h_eff[xi]) + _log1pexp(J_eff[xi, xk] + h_eff[xi]) - _log1pexp(J_eff[xi, xj] + J_eff[xi, xk] + h_eff[xi])) * exact_corr[xj, xk] \
                + (_log1pexp(h_eff[xi]) - _log1pexp(J_eff[xi, xj] + h_eff[xi])) * exact_mean[xj] \
                + (_log1pexp(h_eff[xi]) - _log1pexp(J_eff[xi, xk] + h_eff[xi])) * exact_mean[xk]
            H = H - mi3
    return H


def thermodynamics(J, h, T):
    """Compute energy and specific heat via decimation at temperature T."""
    J_eff = np.asarray(J, dtype=float).copy()
    h_eff = np.asarray(h, dtype=float).copy()
    f = 0.0
    df = 0.0
    dh = np.zeros_like(h_eff)
    dJ = np.zeros_like(J_eff)
    ddf = 0.0
    ddh = np.zeros_like(h_eff)
    ddJ = np.zeros_like(J_eff)
    _, D = decimate_gsp(J)
    for step in range(D.shape[0]):
        i = D[step, 0]
        j = D[step, 1]
        k = D[step, 2]
        if i == -1 or j == -1:
            continue
        h_eff[j] = h_eff[j] - T * _log1pexp(h_eff[i] / T) + T * _log1pexp((J_eff[j, i] + h_eff[i]) / T)
        df = df + _log1pexp(h_eff[i] / T) + 1 / (math.exp(-h_eff[i] / T) + 1) * (dh[i] - h_eff[i] / T)
        dh[j] = dh[j] + math.log((math.exp((h_eff[i] + J_eff[i, j]) / T) + 1) / (math.exp(h_eff[i] / T) + 1)) \
            - 1 / (math.exp(-h_eff[i] / T) + 1) * (dh[i] - h_eff[i] / T) \
            + 1 / (math.exp(-(h_eff[i] + J_eff[i, j]) / T) + 1) * (dh[i] + dJ[i, j] - (h_eff[i] + J_eff[i, j]) / T)
        ddf = ddf + 1 / (math.exp(-h_eff[i] / T) + 1) * ddh[i] + math.exp(-h_eff[i] / T) / (T * (math.exp(-h_eff[i] / T) + 1) ** 2) * ((dh[i] - h_eff[i] / T) ** 2)
        ddh[j] = ddh[j] + 1 / (math.exp(-(h_eff[i] + J_eff[i, j]) / T) + 1) * (ddh[i] + ddJ[i, j]) \
            - 1 / (math.exp(-h_eff[i] / T) + 1) * ddh[i] \
            - math.exp(-h_eff[i] / T) / ((math.exp(-h_eff[i] / T) + 1) ** 2) * ((dh[i] - h_eff[i] / T) ** 2) / T \
            + math.exp(-(h_eff[i] + J_eff[i, j]) / T) / ((math.exp(-(h_eff[i] + J_eff[i, j]) / T) + 1) ** 2) * ((dh[i] + dJ[i, j] - (h_eff[i] + J_eff[i, j]) / T) ** 2) / T
        if k != -1:
            h_eff[k] = h_eff[k] - T * _log1pexp(h_eff[i] / T) + T * _log1pexp((J_eff[i, k] + h_eff[i]) / T)
            J_eff[j, k] = J_eff[j, k] + T * _log1pexp(h_eff[i] / T) - T * _log1pexp((J_eff[i, j] + h_eff[i]) / T) - T * _log1pexp((J_eff[i, k] + h_eff[i]) / T) + T * _log1pexp((J_eff[i, j] + J_eff[i, k] + h_eff[i]) / T)
            J_eff[k, j] = J_eff[j, k]
            dh[k] = dh[k] + math.log((math.exp((h_eff[i] + J_eff[i, k]) / T) + 1) / (math.exp(h_eff[i] / T) + 1)) \
                - 1 / (math.exp(-h_eff[i] / T) + 1) * (dh[i] - h_eff[i] / T) \
                + 1 / (math.exp(-(h_eff[i] + J_eff[i, k]) / T) + 1) * (dh[i] + dJ[i, k] - (h_eff[i] + J_eff[i, k]) / T)
            dJ[j, k] = dJ[j, k] + math.log(((math.exp((h_eff[i] + J_eff[i, k] + J_eff[i, j]) / T) + 1) * (math.exp(h_eff[i] / T) + 1)) / ((math.exp((h_eff[i] + J_eff[i, k]) / T) + 1) * (math.exp((h_eff[i] + J_eff[i, j]) / T) + 1))) \
                + 1 / (math.exp(-h_eff[i] / T) + 1) * (dh[i] - h_eff[i] / T) \
                - 1 / (math.exp(-(h_eff[i] + J_eff[i, j]) / T) + 1) * (dh[i] + dJ[i, j] - (h_eff[i] + J_eff[i, j]) / T) \
                - 1 / (math.exp(-(h_eff[i] + J_eff[i, k]) / T) + 1) * (dh[i] + dJ[i, k] - (h_eff[i] + J_eff[i, k]) / T) \
                + 1 / (math.exp(-(h_eff[i] + J_eff[i, j] + J_eff[i, k]) / T) + 1) * (dh[i] + dJ[i, j] + dJ[i, k] - (h_eff[i] + J_eff[i, j] + J_eff[i, k]) / T)
            dJ[k, j] = dJ[j, k]
            ddh[k] = ddh[k] + 1 / (math.exp(-(h_eff[i] + J_eff[i, k]) / T) + 1) * (ddh[i] + ddJ[i, k]) \
                - 1 / (math.exp(-h_eff[i] / T) + 1) * ddh[i] \
                - math.exp(-h_eff[i] / T) / ((math.exp(-h_eff[i] / T) + 1) ** 2) * ((dh[i] - h_eff[i] / T) ** 2) / T \
                + math.exp(-(h_eff[i] + J_eff[i, k]) / T) / ((math.exp(-(h_eff[i] + J_eff[i, k]) / T) + 1) ** 2) * ((dh[i] + dJ[i, k] - (h_eff[i] + J_eff[i, k]) / T) ** 2) / T
            ddJ[j, k] = ddJ[j, k] + 1 / (math.exp(-h_eff[i] / T) + 1) * ddh[i] + math.exp(-h_eff[i] / T) / ((math.exp(-h_eff[i] / T) + 1) ** 2) * ((dh[i] - h_eff[i] / T) ** 2) / T \
                - 1 / (math.exp(-(h_eff[i] + J_eff[i, j]) / T) + 1) * (ddh[i] + ddJ[i, j]) - math.exp(-(h_eff[i] + J_eff[i, j]) / T) / ((math.exp(-(h_eff[i] + J_eff[i, j]) / T) + 1) ** 2) * ((dh[i] + dJ[i, j] - (h_eff[i] + J_eff[i, j]) / T) ** 2) / T \
                - 1 / (math.exp(-(h_eff[i] + J_eff[i, k]) / T) + 1) * (ddh[i] + ddJ[i, k]) - math.exp(-(h_eff[i] + J_eff[i, k]) / T) / ((math.exp(-(h_eff[i] + J_eff[i, k]) / T) + 1) ** 2) * ((dh[i] + dJ[i, k] - (h_eff[i] + J_eff[i, k]) / T) ** 2) / T \
                + 1 / (math.exp(-(h_eff[i] + J_eff[i, j] + J_eff[i, k]) / T) + 1) * (ddh[i] + ddJ[i, j] + ddJ[i, k]) + math.exp(-(h_eff[i] + J_eff[i, j] + J_eff[i, k]) / T) / ((math.exp(-(h_eff[i] + J_eff[i, j] + J_eff[i, k]) / T) + 1) ** 2) * ((dh[i] + dJ[i, j] + dJ[i, k] - (h_eff[i] + J_eff[i, j] + J_eff[i, k]) / T) ** 2) / T
            ddJ[k, j] = ddJ[j, k]
    df = df + _log1pexp(h_eff[j] / T) + 1 / (math.exp(-h_eff[j] / T) + 1) * (dh[j] - h_eff[j] / T)
    ddf = ddf + 1 / (math.exp(-h_eff[j] / T) + 1) * ddh[j] + math.exp(-h_eff[j] / T) / (T * (math.exp(-h_eff[j] / T) + 1) ** 2) * ((dh[j] - h_eff[j] / T) ** 2)
    for i in range(len(h)):
        f = f + T * _log1pexp(h_eff[i] / T)
    energy = -f + T * df
    specific_heat = T * ddf
    return energy, specific_heat, D


def _exact_moments(J, h, kT=1.0):
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    num_spins = len(h)
    z = 0.0
    mean_spin = np.zeros(num_spins)
    corr = np.zeros((num_spins, num_spins))
    threepoint = np.zeros((num_spins, num_spins, num_spins))
    for i in range(2 ** num_spins):
        spin_set = _int_to_binary_vector(i, num_spins)
        weight = math.exp(-energy_ising(spin_set, J, h) / kT)
        z += weight
        mean_spin += weight * spin_set
        corr += weight * np.outer(spin_set, spin_set)
        aa, bb, cc = np.meshgrid(spin_set, spin_set, spin_set, indexing="ij")
        threepoint += weight * (aa * bb * cc)
    mean_spin /= z
    corr /= z
    threepoint /= z
    return mean_spin, corr, threepoint, z


def _correlations_gsp_core(J, h, D, Jcon, connum, with_trip):
    """Core GSP correlation routine (0/1 spins).

    Performs decimation down to one node, then back-propagates
    magnetizations, correlations, and derivatives. Optionally computes
    triplets for edge-node combinations.
    """
    n = len(h)
    # Decimate nodes of degree 1 or 2 down to a single node
    J_eff = J.copy()
    h_eff = h.copy()
    c_val = 0.0  # accumulated log-partition contributions
    for t in range(D.shape[0]):
        i, j, k = D[t]
        if i == -1 or j == -1:
            continue
        c_val += _log1pexp(h_eff[i])
        h_eff[j] = h_eff[j] - _log1pexp(h_eff[i]) + _log1pexp(J_eff[i, j] + h_eff[i])
        if k != -1:
            h_eff[k] = h_eff[k] - _log1pexp(h_eff[i]) + _log1pexp(J_eff[i, k] + h_eff[i])
            J_eff[j, k] = J_eff[j, k] + _log1pexp(h_eff[i]) - _log1pexp(J_eff[i, j] + h_eff[i]) - _log1pexp(J_eff[i, k] + h_eff[i]) + _log1pexp(J_eff[i, j] + J_eff[i, k] + h_eff[i])
            J_eff[k, j] = J_eff[j, k]

    # Allocate magnetizations, correlations, and derivative buffers
    m = np.zeros(n)
    C = np.zeros((n, n))
    dm_dh = np.zeros((n, n))
    dC_dh = np.zeros((connum, n))
    dh_dh = np.eye(n)
    dJ_dh = np.zeros((connum, n))
    dh_dJ = np.zeros((n, connum))
    Jpair = np.zeros((connum, connum))
    Trip = np.zeros((connum, n)) if with_trip else None

    for ind1 in range(n):
        for ind2 in range(n):
            if Jcon[ind1, ind2] != -1:
                idx = Jcon[ind1, ind2]
                Jpair[idx, idx] = 1

    # Compute quantities for the final node after decimation
    valid_rows = D[D[:, 1] != -1]
    i0 = valid_rows[-1, 1]
    m[i0] = 1 / (1 + math.exp(-h_eff[i0]))
    C[i0, i0] = m[i0]
    dm_dh[i0, i0] = math.exp(-h_eff[i0]) / (1 + math.exp(-h_eff[i0])) ** 2
    Z = math.exp(c_val) * (math.exp(h_eff[i0]) + 1)

    # Back-propagate magnetizations and correlations in reverse order
    for t_j in range(D.shape[0] - 1, -1, -1):
        j1, j2, j3 = D[t_j]
        if j1 == -1 or j2 == -1:
            continue

        if j3 == -1:
            m[j1] = (1 - m[j2]) / (1 + math.exp(-h_eff[j1])) + m[j2] / (1 + math.exp(-J_eff[j1, j2] - h_eff[j1]))
            C[j1, j1] = m[j1]
            C[j1, j2] = m[j2] / (1 + math.exp(-J_eff[j1, j2] - h_eff[j1]))
            C[j2, j1] = C[j1, j2]

            dh_dh[j2, j1] = -1 / (1 + math.exp(-h_eff[j1])) + 1 / (1 + math.exp(-J_eff[j1, j2] - h_eff[j1]))
            dh_dJ[j2, Jcon[j1, j2]] = 1 / (1 + math.exp(-J_eff[j1, j2] - h_eff[j1]))
            dh_dJ[j2, Jcon[j2, j1]] = dh_dJ[j2, Jcon[j1, j2]]

            dh_dh[i0, j1] = dh_dh[i0, j2] * dh_dh[j2, j1]
            dh_dJ[i0, Jcon[j1, j2]] = dh_dh[i0, j2] * dh_dJ[j2, Jcon[j1, j2]]
            dh_dJ[i0, Jcon[j2, j1]] = dh_dJ[i0, Jcon[j1, j2]]

            for t_i in range(D.shape[0] - 1, t_j, -1):
                i1, i2, i3 = D[t_i]
                if i1 == -1 or i2 == -1:
                    continue
                dh_dh[i1, j1] = dh_dh[i1, j2] * dh_dh[j2, j1]
                dh_dJ[i1, Jcon[j1, j2]] = dh_dh[i1, j2] * dh_dJ[j2, Jcon[j1, j2]]
                dh_dJ[i1, Jcon[j2, j1]] = dh_dJ[i1, Jcon[j1, j2]]

                dJ_dh[Jcon[i1, i2], j1] = dJ_dh[Jcon[i1, i2], j2] * dh_dh[j2, j1]
                dJ_dh[Jcon[i2, i1], j1] = dJ_dh[Jcon[i1, i2], j1]

                Jpair[Jcon[i1, i2], Jcon[j1, j2]] = dJ_dh[Jcon[i1, i2], j2] * dh_dJ[j2, Jcon[j1, j2]]
                Jpair[Jcon[i2, i1], Jcon[j1, j2]] = Jpair[Jcon[i1, i2], Jcon[j1, j2]]
                Jpair[Jcon[i1, i2], Jcon[j2, j1]] = Jpair[Jcon[i1, i2], Jcon[j1, j2]]
                Jpair[Jcon[i2, i1], Jcon[j2, j1]] = Jpair[Jcon[i1, i2], Jcon[j1, j2]]

                if i3 != -1:
                    dJ_dh[Jcon[i1, i3], j1] = dJ_dh[Jcon[i1, i3], j2] * dh_dh[j2, j1]
                    dJ_dh[Jcon[i3, i1], j1] = dJ_dh[Jcon[i1, i3], j1]

                    Jpair[Jcon[i1, i3], Jcon[j1, j2]] = dJ_dh[Jcon[i1, i3], j2] * dh_dJ[j2, Jcon[j1, j2]]
                    Jpair[Jcon[i3, i1], Jcon[j1, j2]] = Jpair[Jcon[i1, i3], Jcon[j1, j2]]
                    Jpair[Jcon[i1, i3], Jcon[j2, j1]] = Jpair[Jcon[i1, i3], Jcon[j1, j2]]
                    Jpair[Jcon[i3, i1], Jcon[j2, j1]] = Jpair[Jcon[i1, i3], Jcon[j1, j2]]
        else:
            m[j1] = (1 - m[j2] - m[j3] + C[j2, j3]) / (1 + math.exp(-h_eff[j1])) \
                + (m[j2] - C[j2, j3]) / (1 + math.exp(-J_eff[j1, j2] - h_eff[j1])) \
                + (m[j3] - C[j2, j3]) / (1 + math.exp(-J_eff[j1, j3] - h_eff[j1])) \
                + C[j2, j3] / (1 + math.exp(-J_eff[j1, j2] - J_eff[j1, j3] - h_eff[j1]))
            C[j1, j1] = m[j1]

            C[j1, j2] = (m[j2] - C[j2, j3]) / (1 + math.exp(-J_eff[j1, j2] - h_eff[j1])) \
                + C[j2, j3] / (1 + math.exp(-J_eff[j1, j2] - J_eff[j1, j3] - h_eff[j1]))
            C[j2, j1] = C[j1, j2]

            C[j1, j3] = (m[j3] - C[j2, j3]) / (1 + math.exp(-J_eff[j1, j3] - h_eff[j1])) \
                + C[j2, j3] / (1 + math.exp(-J_eff[j1, j2] - J_eff[j1, j3] - h_eff[j1]))
            C[j3, j1] = C[j1, j3]

            dh_dh[j2, j1] = -1 / (1 + math.exp(-h_eff[j1])) + 1 / (1 + math.exp(-J_eff[j1, j2] - h_eff[j1]))
            dh_dJ[j2, Jcon[j1, j2]] = 1 / (1 + math.exp(-J_eff[j1, j2] - h_eff[j1]))
            dh_dJ[j2, Jcon[j2, j1]] = dh_dJ[j2, Jcon[j1, j2]]

            dh_dh[j3, j1] = -1 / (1 + math.exp(-h_eff[j1])) + 1 / (1 + math.exp(-J_eff[j1, j3] - h_eff[j1]))
            dh_dJ[j3, Jcon[j1, j3]] = 1 / (1 + math.exp(-J_eff[j1, j3] - h_eff[j1]))
            dh_dJ[j3, Jcon[j3, j1]] = dh_dJ[j3, Jcon[j1, j3]]

            dJ_dh[Jcon[j2, j3], j1] = 1 / (1 + math.exp(-h_eff[j1])) - 1 / (1 + math.exp(-J_eff[j1, j2] - h_eff[j1])) \
                - 1 / (1 + math.exp(-J_eff[j1, j3] - h_eff[j1])) + 1 / (1 + math.exp(-J_eff[j1, j2] - J_eff[j1, j3] - h_eff[j1]))
            dJ_dh[Jcon[j3, j2], j1] = dJ_dh[Jcon[j2, j3], j1]

            Jpair[Jcon[j2, j3], Jcon[j1, j2]] = -1 / (1 + math.exp(-J_eff[j1, j2] - h_eff[j1])) \
                + 1 / (1 + math.exp(-J_eff[j1, j2] - J_eff[j1, j3] - h_eff[j1]))
            Jpair[Jcon[j2, j3], Jcon[j2, j1]] = Jpair[Jcon[j2, j3], Jcon[j1, j2]]
            Jpair[Jcon[j3, j2], Jcon[j1, j2]] = Jpair[Jcon[j2, j3], Jcon[j1, j2]]
            Jpair[Jcon[j3, j2], Jcon[j2, j1]] = Jpair[Jcon[j2, j3], Jcon[j1, j2]]

            Jpair[Jcon[j2, j3], Jcon[j1, j3]] = -1 / (1 + math.exp(-J_eff[j1, j3] - h_eff[j1])) \
                + 1 / (1 + math.exp(-J_eff[j1, j2] - J_eff[j1, j3] - h_eff[j1]))
            Jpair[Jcon[j2, j3], Jcon[j3, j1]] = Jpair[Jcon[j2, j3], Jcon[j1, j3]]
            Jpair[Jcon[j3, j2], Jcon[j1, j3]] = Jpair[Jcon[j2, j3], Jcon[j1, j3]]
            Jpair[Jcon[j3, j2], Jcon[j3, j1]] = Jpair[Jcon[j2, j3], Jcon[j1, j3]]

            dh_dh[i0, j1] = dh_dh[i0, j2] * dh_dh[j2, j1] + dh_dh[i0, j3] * dh_dh[j3, j1] \
                + dh_dJ[i0, Jcon[j2, j3]] * dJ_dh[Jcon[j2, j3], j1]
            dh_dJ[i0, Jcon[j1, j2]] = dh_dh[i0, j2] * dh_dJ[j2, Jcon[j1, j2]] \
                + dh_dJ[i0, Jcon[j2, j3]] * Jpair[Jcon[j2, j3], Jcon[j1, j2]]
            dh_dJ[i0, Jcon[j2, j1]] = dh_dJ[i0, Jcon[j1, j2]]
            dh_dJ[i0, Jcon[j1, j3]] = dh_dh[i0, j3] * dh_dJ[j3, Jcon[j1, j3]] \
                + dh_dJ[i0, Jcon[j2, j3]] * Jpair[Jcon[j2, j3], Jcon[j1, j3]]
            dh_dJ[i0, Jcon[j3, j1]] = dh_dJ[i0, Jcon[j1, j3]]

            for t_i in range(D.shape[0] - 1, t_j, -1):
                i1, i2, i3 = D[t_i]
                if i1 == -1 or i2 == -1:
                    continue
                dh_dh[i1, j1] = dh_dh[i1, j2] * dh_dh[j2, j1] + dh_dh[i1, j3] * dh_dh[j3, j1] \
                    + dh_dJ[i1, Jcon[j2, j3]] * dJ_dh[Jcon[j2, j3], j1]
                dh_dJ[i1, Jcon[j1, j2]] = dh_dh[i1, j2] * dh_dJ[j2, Jcon[j1, j2]] \
                    + dh_dJ[i1, Jcon[j2, j3]] * Jpair[Jcon[j2, j3], Jcon[j1, j2]]
                dh_dJ[i1, Jcon[j2, j1]] = dh_dJ[i1, Jcon[j1, j2]]
                dh_dJ[i1, Jcon[j1, j3]] = dh_dh[i1, j3] * dh_dJ[j3, Jcon[j1, j3]] \
                    + dh_dJ[i1, Jcon[j2, j3]] * Jpair[Jcon[j2, j3], Jcon[j1, j3]]
                dh_dJ[i1, Jcon[j3, j1]] = dh_dJ[i1, Jcon[j1, j3]]

                dJ_dh[Jcon[i1, i2], j1] = dJ_dh[Jcon[i1, i2], j2] * dh_dh[j2, j1] + dJ_dh[Jcon[i1, i2], j3] * dh_dh[j3, j1] \
                    + dJ_dh[Jcon[j2, j3], j1] * Jpair[Jcon[i1, i2], Jcon[j2, j3]]
                dJ_dh[Jcon[i2, i1], j1] = dJ_dh[Jcon[i1, i2], j1]

                Jpair[Jcon[i1, i2], Jcon[j1, j2]] = dJ_dh[Jcon[i1, i2], j2] * dh_dJ[j2, Jcon[j1, j2]] \
                    + Jpair[Jcon[i1, i2], Jcon[j2, j3]] * Jpair[Jcon[j2, j3], Jcon[j1, j2]]
                Jpair[Jcon[i1, i2], Jcon[j2, j1]] = Jpair[Jcon[i1, i2], Jcon[j1, j2]]
                Jpair[Jcon[i2, i1], Jcon[j1, j2]] = Jpair[Jcon[i1, i2], Jcon[j1, j2]]
                Jpair[Jcon[i2, i1], Jcon[j2, j1]] = Jpair[Jcon[i1, i2], Jcon[j1, j2]]

                Jpair[Jcon[i1, i2], Jcon[j1, j3]] = dJ_dh[Jcon[i1, i2], j3] * dh_dJ[j3, Jcon[j1, j3]] \
                    + Jpair[Jcon[i1, i2], Jcon[j2, j3]] * Jpair[Jcon[j2, j3], Jcon[j1, j3]]
                Jpair[Jcon[i1, i2], Jcon[j3, j1]] = Jpair[Jcon[i1, i2], Jcon[j1, j3]]
                Jpair[Jcon[i2, i1], Jcon[j1, j3]] = Jpair[Jcon[i1, i2], Jcon[j1, j3]]
                Jpair[Jcon[i2, i1], Jcon[j3, j1]] = Jpair[Jcon[i1, i2], Jcon[j1, j3]]

                if i3 != -1:
                    dJ_dh[Jcon[i1, i3], j1] = dJ_dh[Jcon[i1, i3], j2] * dh_dh[j2, j1] + dJ_dh[Jcon[i1, i3], j3] * dh_dh[j3, j1] \
                        + dJ_dh[Jcon[j2, j3], j1] * Jpair[Jcon[i1, i3], Jcon[j2, j3]]
                    dJ_dh[Jcon[i3, i1], j1] = dJ_dh[Jcon[i1, i3], j1]

                    Jpair[Jcon[i1, i3], Jcon[j1, j2]] = dJ_dh[Jcon[i1, i3], j2] * dh_dJ[j2, Jcon[j1, j2]] \
                        + Jpair[Jcon[i1, i3], Jcon[j2, j3]] * Jpair[Jcon[j2, j3], Jcon[j1, j2]]
                    Jpair[Jcon[i1, i3], Jcon[j2, j1]] = Jpair[Jcon[i1, i3], Jcon[j1, j2]]
                    Jpair[Jcon[i3, i1], Jcon[j1, j2]] = Jpair[Jcon[i1, i3], Jcon[j1, j2]]
                    Jpair[Jcon[i3, i1], Jcon[j2, j1]] = Jpair[Jcon[i1, i3], Jcon[j1, j2]]

                    Jpair[Jcon[i1, i3], Jcon[j1, j3]] = dJ_dh[Jcon[i1, i3], j3] * dh_dJ[j3, Jcon[j1, j3]] \
                        + Jpair[Jcon[i1, i3], Jcon[j2, j3]] * Jpair[Jcon[j2, j3], Jcon[j1, j3]]
                    Jpair[Jcon[i1, i3], Jcon[j3, j1]] = Jpair[Jcon[i1, i3], Jcon[j1, j3]]
                    Jpair[Jcon[i3, i1], Jcon[j1, j3]] = Jpair[Jcon[i1, i3], Jcon[j1, j3]]
                    Jpair[Jcon[i3, i1], Jcon[j3, j1]] = Jpair[Jcon[i1, i3], Jcon[j1, j3]]

        dm_dh[i0, j1] = dm_dh[i0, i0] * dh_dh[i0, j1]
        dm_dh[j1, i0] = dm_dh[i0, j1]

        for t_i in range(D.shape[0] - 1, t_j, -1):
            i1, i2, i3 = D[t_i]
            if i1 == -1 or i2 == -1:
                continue
            if i3 == -1:
                l_h = 1 / (1 + math.exp(-h_eff[i1]))
                l_hJ = 1 / (1 + math.exp(-J_eff[i1, i2] - h_eff[i1]))
                dl_h = math.exp(-h_eff[i1]) / (1 + math.exp(-h_eff[i1])) ** 2
                dl_hJ = math.exp(-J_eff[i1, i2] - h_eff[i1]) / (1 + math.exp(-J_eff[i1, i2] - h_eff[i1])) ** 2

                dm_dh[i1, j1] = dl_h * (1 - m[i2]) * dh_dh[i1, j1] + dl_hJ * m[i2] * (dh_dh[i1, j1] + dJ_dh[Jcon[i1, i2], j1]) \
                    + (-l_h + l_hJ) * dm_dh[i2, j1]
                dm_dh[j1, i1] = dm_dh[i1, j1]

                dC_dh[Jcon[i1, i2], j1] = dl_hJ * m[i2] * (dh_dh[i1, j1] + dJ_dh[Jcon[i1, i2], j1]) + l_hJ * dm_dh[i2, j1]
                dC_dh[Jcon[i2, i1], j1] = dC_dh[Jcon[i1, i2], j1]

                if with_trip:
                    Trip[Jcon[i1, i2], j1] = dC_dh[Jcon[i1, i2], j1] - m[i1] * dm_dh[i2, j1] - m[i2] * dm_dh[i1, j1] \
                        + m[i1] * (dm_dh[i2, j1] + m[i2] * m[j1]) + m[i2] * (dm_dh[i1, j1] + m[i1] * m[j1]) \
                        + m[j1] * (dm_dh[i2, i1] + m[i2] * m[i1]) - 2 * m[i1] * m[i2] * m[j1]
                    Trip[Jcon[i2, i1], j1] = Trip[Jcon[i1, i2], j1]
                    if Jcon[i1, j1] != -1:
                        Trip[Jcon[i1, j1], i2] = Trip[Jcon[i1, i2], j1]
                    if Jcon[i2, j1] != -1:
                        Trip[Jcon[i2, j1], i1] = Trip[Jcon[i1, i2], j1]
            else:
                l_h = 1 / (1 + math.exp(-h_eff[i1]))
                l_hJ2 = 1 / (1 + math.exp(-J_eff[i1, i2] - h_eff[i1]))
                l_hJ3 = 1 / (1 + math.exp(-J_eff[i1, i3] - h_eff[i1]))
                l_hJJ = 1 / (1 + math.exp(-J_eff[i1, i2] - J_eff[i1, i3] - h_eff[i1]))
                dl_h = math.exp(-h_eff[i1]) / (1 + math.exp(-h_eff[i1])) ** 2
                dl_hJ2 = math.exp(-J_eff[i1, i2] - h_eff[i1]) / (1 + math.exp(-J_eff[i1, i2] - h_eff[i1])) ** 2
                dl_hJ3 = math.exp(-J_eff[i1, i3] - h_eff[i1]) / (1 + math.exp(-J_eff[i1, i3] - h_eff[i1])) ** 2
                dl_hJJ = math.exp(-J_eff[i1, i2] - J_eff[i1, i3] - h_eff[i1]) / (1 + math.exp(-J_eff[i1, i2] - J_eff[i1, i3] - h_eff[i1])) ** 2

                dm_dh[i1, j1] = dl_h * (1 - m[i2] - m[i3] + C[i2, i3]) * dh_dh[i1, j1] \
                    + dl_hJ2 * (m[i2] - C[i2, i3]) * (dh_dh[i1, j1] + dJ_dh[Jcon[i1, i2], j1]) \
                    + dl_hJ3 * (m[i3] - C[i2, i3]) * (dh_dh[i1, j1] + dJ_dh[Jcon[i1, i3], j1]) \
                    + dl_hJJ * C[i2, i3] * (dh_dh[i1, j1] + dJ_dh[Jcon[i1, i2], j1] + dJ_dh[Jcon[i1, i3], j1]) \
                    + (-l_h + l_hJ2) * dm_dh[i2, j1] + (-l_h + l_hJ3) * dm_dh[i3, j1] \
                    + (l_h - l_hJ2 - l_hJ3 + l_hJJ) * dC_dh[Jcon[i2, i3], j1]
                dm_dh[j1, i1] = dm_dh[i1, j1]

                dC_dh[Jcon[i1, i2], j1] = dl_hJ2 * (m[i2] - C[i2, i3]) * (dh_dh[i1, j1] + dJ_dh[Jcon[i1, i2], j1]) \
                    + dl_hJJ * C[i2, i3] * (dh_dh[i1, j1] + dJ_dh[Jcon[i1, i2], j1] + dJ_dh[Jcon[i1, i3], j1]) \
                    + l_hJ2 * dm_dh[i2, j1] + (-l_hJ2 + l_hJJ) * dC_dh[Jcon[i2, i3], j1]
                dC_dh[Jcon[i2, i1], j1] = dC_dh[Jcon[i1, i2], j1]

                if with_trip:
                    Trip[Jcon[i1, i2], j1] = dC_dh[Jcon[i1, i2], j1] - m[i1] * dm_dh[i2, j1] - m[i2] * dm_dh[i1, j1] \
                        + m[i1] * (dm_dh[i2, j1] + m[i2] * m[j1]) + m[i2] * (dm_dh[i1, j1] + m[i1] * m[j1]) \
                        + m[j1] * (dm_dh[i2, i1] + m[i2] * m[i1]) - 2 * m[i1] * m[i2] * m[j1]
                    Trip[Jcon[i2, i1], j1] = Trip[Jcon[i1, i2], j1]
                    if Jcon[i1, j1] != -1:
                        Trip[Jcon[i1, j1], i2] = Trip[Jcon[i1, i2], j1]
                    if Jcon[i2, j1] != -1:
                        Trip[Jcon[i2, j1], i1] = Trip[Jcon[i1, i2], j1]

                dC_dh[Jcon[i1, i3], j1] = dl_hJ3 * (m[i3] - C[i2, i3]) * (dh_dh[i1, j1] + dJ_dh[Jcon[i1, i3], j1]) \
                    + dl_hJJ * C[i2, i3] * (dh_dh[i1, j1] + dJ_dh[Jcon[i1, i2], j1] + dJ_dh[Jcon[i1, i3], j1]) \
                    + l_hJ3 * dm_dh[i3, j1] + (-l_hJ3 + l_hJJ) * dC_dh[Jcon[i2, i3], j1]
                dC_dh[Jcon[i3, i1], j1] = dC_dh[Jcon[i1, i3], j1]

                if with_trip:
                    Trip[Jcon[i1, i3], j1] = dC_dh[Jcon[i1, i3], j1] - m[i1] * dm_dh[i3, j1] - m[i3] * dm_dh[i1, j1] \
                        + m[i1] * (dm_dh[i3, j1] + m[i3] * m[j1]) + m[i3] * (dm_dh[i1, j1] + m[i1] * m[j1]) \
                        + m[j1] * (dm_dh[i3, i1] + m[i3] * m[i1]) - 2 * m[i1] * m[i3] * m[j1]
                    Trip[Jcon[i3, i1], j1] = Trip[Jcon[i1, i3], j1]
                    if Jcon[i1, j1] != -1:
                        Trip[Jcon[i1, j1], i3] = Trip[Jcon[i1, i3], j1]
                    if Jcon[i3, j1] != -1:
                        Trip[Jcon[i3, j1], i1] = Trip[Jcon[i1, i3], j1]

    C = dm_dh + np.outer(m, m)
    np.fill_diagonal(C, m)
    X = C - np.outer(m, m)
    return m, C, X, Z, Trip, Jcon, dC_dh, dm_dh, dh_dh, dJ_dh, dh_dJ, Jpair


def correlations_gsp_01(J, h):
    """Compute m, C, susceptibility X, and Z for a GSP Ising model (0/1)."""
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    Jdec, D = decimate_gsp(J)
    if np.sum(Jdec) != 0:
        raise ValueError("Network was not able to be decimated!")
    Jcon, connum, _ = jpairs(J, D)
    m, C, X, Z, _, _, _, _, _, _, _, _ = _correlations_gsp_core(J, h, D, Jcon, connum, with_trip=False)
    return m, C, X, Z


def correlations_gsp_02(J, h):
    """Compute m, C, X, Z, and triplets for a GSP Ising model (0/1)."""
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    Jdec, D = decimate_gsp(J)
    if np.sum(Jdec) != 0:
        raise ValueError("Network was not able to be decimated!")
    bJ = np.tril(J) != 0
    x, y = np.where(bJ)
    Jcon = np.full_like(J, -1, dtype=int)
    for i in range(len(x)):
        Jcon[x[i], y[i]] = i
        Jcon[y[i], x[i]] = i
    connum = len(x)
    pairs = np.column_stack([x, y])
    m, C, X, Z, Trip, _, _, _, _, _, _, _ = _correlations_gsp_core(J, h, D, Jcon, connum, with_trip=True)
    return m, C, X, Z, Trip, Jcon, pairs


def correlations_gsp_03(J, h, m, C, Trip, dC_dh, dm_dh, dh_dh, dJ_dh, dh_dJ, Jpair, keep):
    """Variant of correlations_GSP_02 with a 'keep' set for careful decimation."""
    J = np.asarray(J, dtype=float)
    h = np.asarray(h, dtype=float)
    bJ = np.tril(J) != 0
    x, y = np.where(bJ)
    Jcon = np.full_like(J, -1, dtype=int)
    for i in range(len(x)):
        Jcon[x[i], y[i]] = i
        Jcon[y[i], x[i]] = i
    connum = len(x)
    pairs = np.column_stack([x, y])
    Jdec, D1, Jeff, _ = decimate_gsp_carefully(J, h, keep)
    _, D = decimate_gsp(Jeff)
    D = D + np.flipud(D1)
    D = np.flipud(D)
    m, C, X, Z, Trip, _, dC_dh, dm_dh, dh_dh, dJ_dh, dh_dJ, Jpair = _correlations_gsp_core(J, h, D, Jcon, connum, with_trip=True)
    return m, C, X, Z, Trip, Jcon, dC_dh, dm_dh, dh_dh, dJ_dh, dh_dJ, Jpair, pairs
