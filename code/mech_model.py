"""Well-mixed oxygen / cell-cycle / reversible-arrest model for spheroid data.

States (living tissue in units of 1e6 um^3, time in hours):
    R  cycling G1          G  S/G2/M
    Q1, Q2  arrested (G0-like, red in FUCCI); Q2 is only used by the
            two-stage exit variant, otherwise it stays empty
    D  necrotic material
Oxygen is quasi-steady (it equilibrates in seconds, Murphy et al. 2023):
    N = c(t) / (1 + ell * (R + G + eps_q * (Q1 + Q2))),
where c(t) is the external O2 relative to 21% (1 or 2/21).

Rates, all saturating in N with the common threshold K (h = 1):
    k1(N)    = k1bar * N / (K + N)                    G1 -> S
    alpha(N) = abar  * K / (K + N)                    arrest entry (R -> Q1)
    beta(N)  = bbar  * N / (K + N)                    arrest exit (-> R)
    mu(N)    = mu0 + muh * K / (K + N)                death -> D
    D is cleared at rate kD.
Variants
    "finite"   : one arrested stage (exponential exit time, mean 1/beta)
    "twostage" : Q1 -> Q2 -> R, each at rate 2 beta (same mean, Erlang-2 exit)
    "instant"  : entry and exit scaled by a large factor LAMBDA_FAST with the
                 same ratio alpha/beta, i.e. arrest tracks oxygen instantly.
"""
import numpy as np
from scipy.integrate import solve_ivp

K2 = 0.1          # S/G2/M completion rate (1/h), ~10 h green phase (fixed)
EPS_Q = 0.1       # relative O2 consumption of arrested cells (fixed)
C_HYP = 2.0 / 21.0
LAMBDA_FAST = 40.0

PARAM_NAMES = ["k1bar", "mu0", "ell", "K", "abar", "bbar", "muh", "kD"]

# protocols: (initial-condition group, list of (t_start_h, c) pieces)
PROTOCOLS = {
    "normoxia": ("normoxia", [(0.0, 1.0)]),
    "deox_d2": ("normoxia", [(0.0, C_HYP)]),
    "hypoxia": ("hypoxia", [(0.0, C_HYP)]),
    "reox_d2": ("hypoxia", [(0.0, 1.0)]),
    "reox_d4": ("hypoxia", [(0.0, C_HYP), (48.0, 1.0)]),
}


def rates(N, p):
    k1bar, mu0, ell, K, abar, bbar, muh, kD = p
    frac = N / (K + N)
    return (k1bar * frac, abar * (1 - frac), bbar * frac, mu0 + muh * (1 - frac))


def rhs(t, y, p, c, variant):
    R, G, Q1, Q2, D = y
    ell, kD = p[2], p[7]
    N = c / (1.0 + ell * (R + G + EPS_Q * (Q1 + Q2)))
    k1, a, b, mu = rates(N, p)
    if variant == "instant":
        a, b = a * LAMBDA_FAST, b * LAMBDA_FAST
    if variant == "twostage":
        exit_flux, dQ1_out, dQ2 = 2 * b * Q2, 2 * b * Q1, 2 * b * Q1 - 2 * b * Q2 - mu * Q2
    else:
        exit_flux, dQ1_out, dQ2 = b * Q1, b * Q1, -mu * Q2
    dR = 2 * K2 * G - k1 * R - a * R + exit_flux - mu * R
    dG = k1 * R - K2 * G - mu * G
    dQ1 = a * R - dQ1_out - mu * Q1
    dD = mu * (R + G + Q1 + Q2) - kD * D
    return [dR, dG, dQ1, dQ2, dD]


def initial_state(V_living, f_arr, f_nec_unused, p, c0):
    """Arrested share from data; G set quasi-steady given R at the initial oxygen."""
    Q = f_arr * V_living
    cyc = V_living - Q
    # quasi-steady G/R at the initial oxygen (fixed point iteration on N)
    N = c0
    for _ in range(50):
        k1, _, _, mu = rates(N, p)
        g_over_r = k1 / (K2 + mu)
        R = cyc / (1 + g_over_r)
        N = c0 / (1 + p[2] * (cyc + EPS_Q * Q))
    return np.array([R, cyc - R, Q, 0.0, 0.0])


def simulate(p, protocol, ic, t_eval, variant="finite"):
    """Integrate a protocol from day 2 (t = 0 h); t_eval in hours (sorted)."""
    group, pieces = PROTOCOLS[protocol]
    y0 = initial_state(*ic[group], p, pieces[0][1])
    t_end = float(np.max(t_eval))
    bounds = [pc[0] for pc in pieces] + [t_end]
    out = np.full((5, len(t_eval)), np.nan)
    y = y0
    for (t0, c), t1 in zip(pieces, bounds[1:]):
        if t1 <= t0:
            continue
        sol = solve_ivp(rhs, (t0, t1), y, args=(p, c, variant), method="LSODA",
                        rtol=1e-7, atol=1e-9, dense_output=True)
        if not sol.success:
            return None
        mask = (t_eval >= t0) & (t_eval <= t1)
        if mask.any():
            out[:, mask] = sol.sol(t_eval[mask])
        y = sol.y[:, -1]
    return out


def observables(states):
    R, G, Q1, Q2, D = states
    living = R + G + Q1 + Q2
    outer = living + D
    radius = (3.0 * outer * 1e6 / (4.0 * np.pi)) ** (1.0 / 3.0)
    return radius, (Q1 + Q2) / living, D / outer


def oxygen(states, p, c):
    R, G, Q1, Q2, _ = states
    return c / (1.0 + p[2] * (R + G + EPS_Q * (Q1 + Q2)))
