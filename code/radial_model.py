"""Radially structured spheroid model with reversible cell-cycle arrest.

Geometry (Lagrangian shells). The living tissue is split into M shells that
keep their cells (no migration between shells). Dead material from every
shell joins a central necrotic core of volume D (as in Greenspan's model), so
the living shells sit outside the core in their original order:
    r_i = (3 (D + sum_{j<=i} v_j) / (4 pi))^(1/3),  v_j = R_j + E_j + G_j + Q_j.
Units: volumes 1e6 um^3, radii um, time h, oxygen % O2.

Oxygen (quasi-steady, spherical, zero-order consumption as in Greenspan):
    (1/r^2) d/dr (r^2 dp/dr) = a m c_i   in living shell i,  p(R_o) = p_inf,
    c_i = (R + E + G + eps_q Q)_i / v_i; no consumption in the core or in
    anoxic shells (second pass). m is a metabolic adaptation factor,
    dm/dt = (m_target - m)/tau_m, m_target = 1 in 21% and m_h in 2%.

Cell kinetics in shell i at local oxygen p:
    R --k1(p)--> G --k2--> 2R           FUCCI red -> green -> red
    R --alpha(p)--> Q --beta(p)--> R    reversible arrest (red, never green)
    baseline loss mu0 removes cells (shedding/clearance, no core); anoxic
    death mun Kn^4/(Kn^4+p^4) sends volume to the core, cleared at kD in 2%
    and kD_ox in 21% (necrotic material is removed after reoxygenation,
    Murphy et al. 2023).
    k1(p) = k1max p/(K1 + p)
    alpha(p) = amax (1 - s), beta(p) = bmax s,  s = p^hq/(Kq^hq + p^hq)
    mu(p) = mu0 + mun Kn^4/(Kn^4 + p^4)
Variants
    "noQ"    : no arrest state (only slowed G1 progression)
    "instQ"  : arrest in instantaneous equilibrium with oxygen: the red pool
               X = R + Q is split as Q = X alpha/(alpha+beta) at every time
               (exact quasi-steady reduction; ratio amax/bmax is the parameter)
    "finQ"   : finite alpha, beta (exponential exit time)
    "delayQ" : finite, Erlang-2 entry R -2a-> E -2a-> Q with E -beta-> R

Observables (definitions of Murphy et al. 2023 / Browning et al. 2021):
    R_o outer radius; R_n necrotic radius;
    R_i distance from the periphery at which the green density G/v falls below
        20% of its maximum (R_i = R_n if it never does);
    R_p outer radius of the region with p < p_pim, the effective pimonidazole
        detection threshold (default P_PIM = 1.3% O2, ~10 mmHg; may be fitted).

All protocols are integrated together (batched classical RK4, step DT).
"""
import os

import numpy as np

K2 = 0.1
EPS_Q = 0.5
P_PIM = 1.3
HILL = 2.0
HILL_N = 4.0
GREEN_THRESHOLD = 0.2
FOUR_PI_3 = 4.0 * np.pi / 3.0
DT = 0.5
P21, P2 = 21.0, 2.0
T0_DAY = 1.0          # simulations start at day 1 (t = 0)
# v7 numerics (defaults reproduce v1-v6):
#  RADIAL_PIM_ANALYTIC=1  R_pim from the exact piecewise oxygen profile instead of shell midpoints
#  RADIAL_SHELL_LAYOUT=thickness  initial shells of equal thickness (finer centre) instead of equal volume
PIM_ANALYTIC = os.environ.get("RADIAL_PIM_ANALYTIC", "0") == "1"
SHELL_LAYOUT = os.environ.get("RADIAL_SHELL_LAYOUT", "volume")

PARAM_NAMES = ["k1max", "K1", "a", "amax", "bmax", "Kq", "hq", "mu0", "mun", "Kn", "kD",
               "kD_ox", "m_h", "tau_m", "R1_norm", "R1_hyp", "p_pim"]

# protocol: (initial group, switch time in h since day 1 or None, p before, p after)
PROTOCOLS = {
    "normoxia": ("norm", None, P21, P21),
    "deox_d2": ("norm", 24.0, P21, P2),
    "hypoxia": ("hyp", None, P2, P2),
    "reox_d2": ("hyp", 24.0, P2, P21),
    "reox_d4": ("hyp", 72.0, P2, P21),
}


def oxygen(living, cons_frac, D, a_eff):
    """Batched oxygen profile. living, cons_frac: (P, M); D, a_eff: (P,).
    Returns p_mid (P, M), r_in, r_out, and p_inf must be added by caller:
    we return the drop below p_inf at each shell midpoint."""
    V = D[:, None] + np.cumsum(living, axis=1)
    r_out = np.cbrt(V * (1e6 / FOUR_PI_3))
    r_in = np.concatenate([np.cbrt(D * (1e6 / FOUR_PI_3))[:, None], r_out[:, :-1]], axis=1)
    r_mid = np.cbrt(0.5 * (r_in**3 + r_out**3))
    inv_in = np.where(r_in > 1e-9, 1.0 / np.maximum(r_in, 1e-9), 0.0)

    def drops(q):
        cum = np.cumsum(q * (r_out**3 - r_in**3) / 3.0, axis=1)
        A = np.concatenate([np.zeros((q.shape[0], 1)), cum[:, :-1]], axis=1) - q * r_in**3 / 3.0
        drop = A * (inv_in - 1.0 / r_out) + q * (r_out**2 - r_in**2) / 6.0
        tail = np.cumsum(drop[:, ::-1], axis=1)[:, ::-1] - drop   # sum over outer shells
        part = A * (1.0 / r_mid - 1.0 / r_out) + q * (r_out**2 - r_mid**2) / 6.0
        return tail + part
    q = a_eff[:, None] * cons_frac
    return drops(q), q, drops, r_in, r_out, r_mid


def local_p(living, cons_frac, D, a_eff, p_inf, return_q=False):
    """Greenspan-type anoxic region: if oxygen would become negative, the
    innermost shells stop consuming. The number of switched-off shells is the
    smallest one that keeps p >= 0 everywhere (monotone, found by bisection);
    switched-off shells then share the oxygen level of the consuming edge."""
    dmid, q, drops, r_in, r_out, r_mid = oxygen(living, cons_frac, D, a_eff)
    p = p_inf[:, None] - dmid
    need = p.min(axis=1) < 0
    if need.any():
        Pn, M = p.shape
        idx = np.arange(M)[None, :]
        # Plateau oxygen when shells 0..n-1 do not consume, for every n at once.
        # With all shells consuming, drop_k = A_k (1/r_in_k - 1/r_out_k) + ...;
        # removing the inner n shells lowers every A_k (k >= n) by cum_{n-1},
        # and the sum of (1/r_in_k - 1/r_out_k) over k >= n telescopes, so
        #   plateau(n) = p_inf - T_n + cum_{n-1} (1/r_in_n - 1/R_o),
        # with T_n the total drop from shell n outwards (all consuming).
        vol = q * (r_out**3 - r_in**3) / 3.0
        cum = np.cumsum(vol, axis=1)
        cum_prev = np.concatenate([np.zeros((Pn, 1)), cum[:, :-1]], axis=1)
        A = cum_prev - q * r_in**3 / 3.0
        inv_in = np.where(r_in > 1e-9, 1.0 / np.maximum(r_in, 1e-9), 0.0)
        drop = A * (inv_in - 1.0 / r_out) + q * (r_out**2 - r_in**2) / 6.0
        T = np.cumsum(drop[:, ::-1], axis=1)[:, ::-1]                 # T_n, n = 0..M-1
        Ro = r_out[:, -1:]
        plateau = p_inf[:, None] - T + cum_prev * np.where(cum_prev > 0, inv_in - 1.0 / Ro, 0.0)
        plateau = np.concatenate([plateau, p_inf[:, None]], axis=1)   # n = M: nothing consumes
        feasible = plateau >= -1e-12
        hi = np.argmax(feasible, axis=1)                             # smallest feasible n
        # p is linear in each shell's consumption: switch off only the fraction of
        # the boundary shell (index hi-1) that makes the anoxic plateau exactly 0
        n = np.maximum(hi - 1, 0)
        rows = np.arange(Pn)
        plat_hi, plat_lo = plateau[rows, hi], plateau[rows, n]
        frac = np.where(hi > 0, plat_hi / np.maximum(plat_hi - plat_lo, 1e-12), 0.0)
        frac = np.clip(frac, 0.0, 1.0)  # consuming fraction of the boundary shell
        q_eff = np.where(idx < n[:, None], 0.0, q)
        q_eff = np.where(idx == n[:, None], q * frac[:, None], q_eff)
        q_eff = np.where(need[:, None], q_eff, q)
        p = p_inf[:, None] - drops(q_eff)
    if return_q:
        return np.clip(p, 0.0, None), r_in, r_out, r_mid, (q_eff if need.any() else q)
    return np.clip(p, 0.0, None), r_in, r_out, r_mid


def pim_radius_exact(living, cons_frac, D, a_eff, p_inf, level):
    """Outer radius of the region p < level from the exact piecewise solution
    p(r) = p_inf - tail_i - A_i (1/r - 1/r_out_i) - q_i (r_out_i^2 - r^2)/6 in shell i
    (p increases with r, so the region is a ball). Shell midpoints are not used."""
    _, r_in, r_out, _, q = local_p(living, cons_frac, D, a_eff, p_inf, return_q=True)
    Pn, M = q.shape
    cum = np.cumsum(q * (r_out**3 - r_in**3) / 3.0, axis=1)
    cum_prev = np.concatenate([np.zeros((Pn, 1)), cum[:, :-1]], axis=1)
    A = cum_prev - q * r_in**3 / 3.0
    inv_in = np.where(r_in > 1e-9, 1.0 / np.maximum(r_in, 1e-9), 0.0)
    drop = A * (inv_in - 1.0 / r_out) + q * (r_out**2 - r_in**2) / 6.0
    tail = np.cumsum(drop[:, ::-1], axis=1)[:, ::-1] - drop
    p_out = p_inf[:, None] - tail              # p at the outer edge of each shell
    p_in = p_out - drop                        # p at the inner edge
    out = np.zeros(Pn)
    for i in range(Pn):
        if p_in[i, 0] >= level:               # nowhere below the threshold
            continue
        k = int(np.searchsorted(p_out[i], level, side="left"))  # first shell whose outer edge reaches level
        if k >= M:
            out[i] = r_out[i, -1]
            continue
        lo_r, hi_r = max(r_in[i, k], 1e-9), r_out[i, k]
        f = lambda r: (p_inf[i] - tail[i, k] - A[i, k] * (1 / r - 1 / r_out[i, k])
                       - q[i, k] * (r_out[i, k] ** 2 - r**2) / 6.0 - level)
        if f(lo_r) >= 0:
            out[i] = lo_r if r_in[i, k] > 1e-9 else 0.0
            continue
        for _ in range(40):                    # bisection, p monotone in r
            mid = 0.5 * (lo_r + hi_r)
            if f(mid) < 0:
                lo_r = mid
            else:
                hi_r = mid
        out[i] = 0.5 * (lo_r + hi_r)
    return out


def rates(p, th):
    k1 = th["k1max"] * p / (th["K1"] + p)
    h = th["hq"]
    s = p**h / (th["Kq"] ** h + p**h)
    mu_n = th["mun"] * th["Kn"] ** 4 / (th["Kn"] ** 4 + p**4)  # anoxic (necrotic) death
    return k1, th["amax"] * (1 - s), th["bmax"] * s, th["mu0"] + mu_n, mu_n


class Model:
    def __init__(self, variant, M=16, protocols=tuple(PROTOCOLS), g1_stages=1):
        """g1_stages > 1: G1 is an Erlang chain of that many stages with the same
        mean 1/k1 (a minimal G1 duration instead of a memoryless one);
        implemented for the finQ and noQ variants."""
        self.variant, self.M, self.protocols = variant, M, list(protocols)
        self.k = int(g1_stages)
        if self.k > 1 and variant not in ("finQ", "noQ"):
            raise NotImplementedError("Erlang G1 only for finQ / noQ")
        self.P = len(self.protocols)
        self.switch = np.array([PROTOCOLS[p][1] if PROTOCOLS[p][1] is not None else np.inf
                                for p in self.protocols])
        self.p_before = np.array([PROTOCOLS[p][2] for p in self.protocols])
        self.p_after = np.array([PROTOCOLS[p][3] for p in self.protocols])

    def p_inf(self, t):
        return np.where(t < self.switch - 1e-9, self.p_before, self.p_after)

    # state layout: R, E, G, Q (P, M) each; D (P,), m (P,)
    def split_inst(self, X, p, th):
        _, a, b, _, _ = rates(p, th)
        phi = a / (a + b + 1e-30)
        return X * (1 - phi), X * phi

    def derivs(self, t, S, th, t_env=None):
        """t_env: time at which the external oxygen is read (the step midpoint
        in run(), so that the environment is constant within each RK4 step)."""
        R, E, G, Q, D, m = S
        v = self.variant
        pin = self.p_inf(t if t_env is None else t_env)
        m_target = np.where(pin > 10, 1.0, th["m_h"])
        kD = np.where(pin > 10, th.get("kD_ox", th["kD"]), th["kD"])
        if v == "instQ":
            X = R  # red pool
            living = X + G + 1e-12
            p, *_ = local_p(living, (X + G) / living, D, th["a"] * m, pin)
            Rr, Qq = self.split_inst(X, p, th)
            p, *_ = local_p(living, (Rr + G + EPS_Q * Qq) / living, D, th["a"] * m, pin)
            Rr, Qq = self.split_inst(X, p, th)
            k1, _, _, mu, mu_n = rates(p, th)
            dX = 2 * K2 * G - k1 * Rr - mu * X
            dG = k1 * Rr - K2 * G - mu * G
            dD = np.sum(mu_n * living, axis=1) - kD * D
            z = np.zeros_like(R)
            return (dX, z, dG, z, dD, (m_target - m) / th["tau_m"])
        if self.k > 1:  # Erlang G1: R has shape (P, M, k)
            Rt = R.sum(-1)
            living = Rt + E + G + Q + 1e-12
            p, *_ = local_p(living, (Rt + E + G + EPS_Q * Q) / living, D, th["a"] * m, pin)
            k1, a, b, mu, mu_n = rates(p, th)
            if v == "noQ":
                a = 0.0 * a
            flow = (self.k * k1)[..., None] * R
            dR = -flow - (a + mu)[..., None] * R
            dR[..., 1:] += flow[..., :-1]
            dR[..., 0] += 2 * K2 * G + b * Q
            dG = flow[..., -1] - K2 * G - mu * G
            dQ = a * Rt - b * Q - mu * Q
            dD = np.sum(mu_n * living, axis=1) - kD * D
            return (dR, 0.0 * E, dG, dQ, dD, (m_target - m) / th["tau_m"])
        living = R + E + G + Q + 1e-12
        p, *_ = local_p(living, (R + E + G + EPS_Q * Q) / living, D, th["a"] * m, pin)
        k1, a, b, mu, mu_n = rates(p, th)
        if v == "noQ":
            a = 0.0 * a
        if v == "delayQ":
            to_e, e_to_q, e_back = 2 * a * R, 2 * a * E, b * E
        else:
            to_e, e_to_q, e_back = a * R, a * R, 0.0 * E
        dR = 2 * K2 * G - k1 * R - to_e + e_back + b * Q - mu * R
        dE = (to_e - e_to_q - e_back - mu * E) if v == "delayQ" else 0.0 * E
        dG = k1 * R - K2 * G - mu * G
        dQ = e_to_q - b * Q - mu * Q
        dD = np.sum(mu_n * living, axis=1) - kD * D
        return (dR, dE, dG, dQ, dD, (m_target - m) / th["tau_m"])

    def initial(self, th):
        R1 = np.array([th["R1_norm"] if PROTOCOLS[p][0] == "norm" else th["R1_hyp"]
                       for p in self.protocols])
        V = FOUR_PI_3 * R1**3 / 1e6
        if SHELL_LAYOUT == "thickness":       # equal initial thickness: finer resolution at the centre
            edges = np.linspace(0.0, 1.0, self.M + 1)
            v = V[:, None] * (edges[1:] ** 3 - edges[:-1] ** 3)[None, :]
        else:
            v = np.repeat((V / self.M)[:, None], self.M, axis=1)
        g = th["k1max"] / (th["k1max"] + K2)
        z = np.zeros_like(v)
        # spheroids kept in 2% since seeding start metabolically adapted
        m0 = np.array([1.0 if PROTOCOLS[p][0] == "norm" else th["m_h"] for p in self.protocols])
        R0 = v * (1 - g)
        Q0 = z.copy()
        # optional initial arrest in spheroids grown in 2% (parameter q0_hyp, default 0 = v1-v8):
        # a fraction q0_hyp * (1 - s(p)) of the G1 cells of each shell starts arrested, with p the
        # day-1 oxygen profile of the all-cycling spheroid, so arrest starts where oxygen is low
        q0 = th.get("q0_hyp", 0.0)
        if q0 > 0 and self.variant in ("finQ", "delayQ"):
            hyp = np.array([PROTOCOLS[p][0] != "norm" for p in self.protocols])
            p0, *_ = local_p(v, np.ones_like(v), np.zeros(self.P), th["a"] * m0, self.p_inf(0.0))
            s0 = p0 ** th["hq"] / (th["Kq"] ** th["hq"] + p0 ** th["hq"])
            Q0 = np.where(hyp[:, None], q0 * (1 - s0), 0.0) * R0
            R0 = R0 - Q0
        if self.k > 1:
            R0 = np.repeat((R0 / self.k)[..., None], self.k, axis=-1)
        return [R0, z.copy(), v * g, Q0, np.zeros(self.P), m0]

    def observe(self, S, t, th):
        R, E, G, Q, D, m = S
        if R.ndim == 3:
            R = R.sum(-1)
        pin = self.p_inf(t - 1e-6)  # at a switch time, report the pre-switch state
        if self.variant == "instQ":
            X = R
            living = X + G + 1e-12
            p, *_ = local_p(living, (X + G) / living, D, th["a"] * m, pin)
            R, Q = self.split_inst(X, p, th)
        living = R + E + G + Q + 1e-12
        p, r_in, r_out, r_mid = local_p(living, (R + E + G + EPS_Q * Q) / living, D,
                                        th["a"] * m, pin)
        Ro, Rn = r_out[:, -1], r_in[:, 0]
        Ri = threshold_radius(G / living, r_mid, Ro, Rn, relative=True)
        if PIM_ANALYTIC:
            Rp = pim_radius_exact(living, (R + E + G + EPS_Q * Q) / living, D, th["a"] * m, pin,
                                  th.get("p_pim", P_PIM))
        else:
            Rp = threshold_radius(p, r_mid, Ro, np.zeros_like(Ro), relative=False,
                                  level=th.get("p_pim", P_PIM))
        fQ = (Q + E).sum(1) / living.sum(1)
        return np.stack([Ro, np.maximum(Ri, Rn), Rn, np.minimum(Rp, Ro), fQ], axis=1)

    def run(self, th, t_obs):
        """t_obs: sorted hours since day 1 (multiples of DT). Returns (P, n_t, 5)."""
        S = self.initial(th)
        t_obs = np.asarray(t_obs, float)
        out = np.full((self.P, len(t_obs), 5), np.nan)
        n_steps = int(round(t_obs.max() / DT))
        k_obs = {int(round(t / DT)): i for i, t in enumerate(t_obs)}
        if 0 in k_obs:
            out[:, k_obs[0]] = self.observe(S, 0.0, th)
        for k in range(n_steps):
            t = k * DT
            te = t + DT / 2  # switches fall on step boundaries
            k1 = self.derivs(t, S, th, te)
            k2 = self.derivs(t + DT / 2, [s + DT / 2 * d for s, d in zip(S, k1)], th, te)
            k3 = self.derivs(t + DT / 2, [s + DT / 2 * d for s, d in zip(S, k2)], th, te)
            k4 = self.derivs(t + DT, [s + DT * d for s, d in zip(S, k3)], th, te)
            S = [np.clip(s + DT / 6 * (a + 2 * b + 2 * c + d), 0, None) if i < 5
                 else s + DT / 6 * (a + 2 * b + 2 * c + d)
                 for i, (s, a, b, c, d) in enumerate(zip(S, k1, k2, k3, k4))]
            if not all(np.all(np.isfinite(s)) for s in S):
                return None
            if k + 1 in k_obs:
                out[:, k_obs[k + 1]] = self.observe(S, (k + 1) * DT, th)
        return out


def threshold_radius(field, r_mid, Ro, Rin_default, relative, level=P_PIM):
    """Scan each spheroid from the periphery inwards; return the radius where
    `field` first drops below the threshold (20% of max if relative, P_PIM
    otherwise), interpolated between shell midpoints."""
    Pn, M = field.shape
    thr = GREEN_THRESHOLD * field.max(axis=1) if relative else np.full(Pn, level)
    below = field < thr[:, None]
    out = Rin_default.copy()
    for i in range(Pn):
        idx = np.where(below[i])[0]
        if idx.size == 0:
            continue
        k = idx[-1]
        if k == M - 1:
            out[i] = Ro[i]
            continue
        f0, f1 = field[i, k], field[i, k + 1]
        w = (f1 - thr[i]) / max(f1 - f0, 1e-12)
        out[i] = r_mid[i, k + 1] - w * (r_mid[i, k + 1] - r_mid[i, k])
    return out
