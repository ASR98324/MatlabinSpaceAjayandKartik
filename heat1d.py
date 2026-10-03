"""NumPy solver for the 1D heat equation.

Cartesian   : dT/dt = alpha * d2T/dx2                    + Q(x, t)   on 0 <= x <= L
Cylindrical : dT/dt = alpha * (1/r) d/dr (r dT/dr)        + Q(r, t)   on 0 <= r <= L

Spatial discretization: 2nd-order central differences on a uniform grid.
Time integration:
    'explicit'  Forward Euler (dt limited by stability)
    'implicit'  Backward Euler (unconditionally stable, 1st order in time)
    'cn'        Crank-Nicolson (unconditionally stable, 2nd order in time)
Boundary conditions, one per end (left, right):
    'dirichlet' fixed temperature            T = value
    'neumann'   outward normal gradient      dT/dn = value          (0 = insulated)
    'robin'     convection                   -k dT/dn = h (T - value)
                with coefficient H = h/k passed in bc_coeffs
In cylindrical geometry with the grid starting at r = 0, the left end is the
symmetry axis and is handled automatically (its BC must be 'neumann' with 0).

Batch mode: pass T_init with shape (B, N) and/or source_uniform with shape
(n_times, B) to integrate B independent problems (same grid/BCs, different
initial conditions and sources) in one vectorized run.

Designed to be called from MATLAB (see heat_equation_1d.m and
validate_battery_thermal.m), so inputs are plain arrays rather than callables.
"""

import warnings

import numpy as np


def _laplacian(x, geometry, bc_types, bc_values, bc_coeffs):
    """Tridiagonal operator (lower, diag, upper), boundary vector b, Dirichlet mask.

    Neumann/Robin ends use a ghost node T_ghost = T_neighbor + 2*h*dT/dn.
    """
    N = x.size
    h = x[1] - x[0]

    if geometry == "cartesian":
        aW = np.ones(N)
        aE = np.ones(N)
        axis = False
    elif geometry == "cylindrical":
        if x[0] < 0:
            raise ValueError("cylindrical geometry needs r >= 0")
        axis = x[0] == 0.0
        r = np.where(x > 0, x, 1.0)
        aW = 1.0 - h / (2.0 * r)
        aE = 1.0 + h / (2.0 * r)
        if axis:
            kind, val = str(bc_types[0]).lower(), bc_values[0]
            if kind != "neumann" or val != 0:
                raise ValueError("left end is the symmetry axis (r = 0); use neumann with value 0")
    else:
        raise ValueError(f"Unknown geometry {geometry!r}; use cartesian or cylindrical")

    lower = aW[1:].copy()          # A[i, i-1]
    upper = aE[:-1].copy()         # A[i, i+1]
    diag = -(aW + aE)
    b = np.zeros(N)
    fixed = np.zeros(N, dtype=bool)

    for end in (0, 1):
        i = 0 if end == 0 else N - 1
        if end == 0 and axis:
            # Symmetry axis: (1/r) d/dr(r dT/dr) -> 2 d2T/dr2 = 4 (T1 - T0) / h^2
            diag[0], upper[0] = -4.0, 4.0
            continue
        kind = str(bc_types[end]).lower()
        value, coeff = bc_values[end], bc_coeffs[end]
        wg = aW[0] if end == 0 else aE[-1]      # weight of the ghost node
        if kind in ("neumann", "robin"):
            if end == 0:
                upper[0] += wg
            else:
                lower[-1] += wg
            if kind == "neumann":
                b[i] += wg * 2.0 * h * value
            else:
                diag[i] -= wg * 2.0 * h * coeff
                b[i] += wg * 2.0 * h * coeff * value
        elif kind == "dirichlet":
            fixed[i] = True
        else:
            raise ValueError(f"Unknown BC type {kind!r}")

    lower, diag, upper, b = lower / h**2, diag / h**2, upper / h**2, b / h**2
    # Dirichlet rows are zeroed so those nodes never change
    diag[fixed] = 0.0
    b[fixed] = 0.0
    if fixed[0]:
        upper[0] = 0.0
    if fixed[-1]:
        lower[-1] = 0.0
    return lower, diag, upper, b, fixed


def _matvec(lower, diag, upper, u):
    """Tridiagonal product; u is (N, B)."""
    r = diag[:, None] * u
    r[1:] += lower[:, None] * u[:-1]
    r[:-1] += upper[:, None] * u[1:]
    return r


class _Thomas:
    """Tridiagonal solver; factorized once, then each solve is O(N) per column."""

    def __init__(self, lower, diag, upper):
        N = diag.size
        self.lower = lower
        self.cp = np.zeros(N - 1)
        self.denom = np.zeros(N)
        self.denom[0] = diag[0]
        self.cp[0] = upper[0] / diag[0]
        for i in range(1, N):
            self.denom[i] = diag[i] - lower[i - 1] * self.cp[i - 1]
            if i < N - 1:
                self.cp[i] = upper[i] / self.denom[i]

    def solve(self, d):
        """d is (N, B); every column is solved at once."""
        N = d.shape[0]
        lower, cp, denom = self.lower, self.cp, self.denom
        dp = np.empty_like(d)
        dp[0] = d[0] / denom[0]
        for i in range(1, N):
            dp[i] = (d[i] - lower[i - 1] * dp[i - 1]) / denom[i]
        x = np.empty_like(d)
        x[-1] = dp[-1]
        for i in range(N - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x


def _time_interp(times, table):
    """Linear interpolation of table rows (axis 0) in time, clamped at the ends."""
    def f(t):
        if t <= times[0]:
            return table[0]
        if t >= times[-1]:
            return table[-1]
        k = np.searchsorted(times, t)
        w = (t - times[k - 1]) / (times[k] - times[k - 1])
        return (1.0 - w) * table[k - 1] + w * table[k]
    return f


def _source_fn(source, source_uniform, source_times, N, keep):
    """Build q(t) -> array broadcastable to (N, B)."""
    times = None if source_times is None else np.asarray(source_times, dtype=float).ravel()
    parts = []

    if source is not None:
        src = np.asarray(source, dtype=float)
        if src.size == N:
            q = (src.ravel() * keep)[:, None]
            parts.append(lambda t, q=q: q)
        else:
            nt = times.size
            if src.shape != (nt, N):
                src = src.T if src.shape == (N, nt) else src.reshape(nt, N)
            f = _time_interp(times, src * keep)
            parts.append(lambda t, f=f: f(t)[:, None])

    if source_uniform is not None:
        su = np.asarray(source_uniform, dtype=float)
        nt = times.size
        if su.ndim == 1:
            su = su.reshape(nt, 1)
        elif su.shape[0] != nt and su.shape[1] == nt:
            su = su.T
        f = _time_interp(times, su)
        kk = keep[:, None]
        parts.append(lambda t, f=f: kk * f(t)[None, :])

    if not parts:
        zero = np.zeros((N, 1))
        return lambda t: zero
    if len(parts) == 1:
        return parts[0]
    return lambda t: sum(p(t) for p in parts)


def solve(L, N, alpha, t_final, method="cn", dt=None, geometry="cartesian",
          bc_types=("dirichlet", "dirichlet"), bc_values=(0.0, 0.0), bc_coeffs=(0.0, 0.0),
          T_init=None, source=None, source_uniform=None, source_times=None,
          save_every=1, probe_index=None, store_field=True):
    """Integrate the 1D heat equation.

    Parameters
    ----------
    L, N, alpha, t_final : domain length (or radius), grid nodes, diffusivity, end time
    method        : 'explicit', 'implicit' or 'cn'
    dt            : time step (None -> automatic)
    geometry      : 'cartesian' or 'cylindrical' (x is then the radius)
    bc_types      : (left, right), each 'dirichlet', 'neumann' or 'robin'
    bc_values     : (left, right) temperature, outward gradient, or ambient temperature
    bc_coeffs     : (left, right) H = h/k for 'robin' ends (ignored otherwise)
    T_init        : (N,) initial temperature, or (B, N) for a batch (default zeros)
    source        : None, (N,) constant Q, or (n_times, N) table over source_times
    source_uniform: spatially uniform Q over source_times, (n_times,) or (n_times, B)
    save_every    : store output every this many steps
    probe_index   : node index (0-based, negative allowed) to record at every saved step
    store_field   : store full T snapshots (set False for big batches)

    Returns
    -------
    dict with 'x', 't', 'dt', 'n_steps', 'method' and, when requested,
    'T'     : (n_saved, N) or (n_saved, B, N)
    'probe' : (n_saved,) or (n_saved, B)
    """
    N = int(N)
    L, alpha, t_final = float(L), float(alpha), float(t_final)
    method = str(method).lower()
    geometry = str(geometry).lower()
    bc_types = [str(s) for s in bc_types]
    bc_values = [float(v) for v in np.asarray(bc_values, dtype=float).ravel()]
    bc_coeffs = [float(v) for v in np.asarray(bc_coeffs, dtype=float).ravel()]
    save_every = max(int(save_every), 1)

    x = np.linspace(0.0, L, N)
    h = x[1] - x[0]
    lower, diag, upper, b, fixed = _laplacian(x, geometry, bc_types, bc_values, bc_coeffs)
    keep = (~fixed).astype(float)
    q = _source_fn(source, source_uniform, source_times, N, keep)

    # Initial condition as (N, B)
    if T_init is None:
        u = np.zeros((N, 1))
    else:
        T0 = np.asarray(T_init, dtype=float)
        if T0.size == N:
            u = T0.reshape(N, 1).copy()
        elif T0.ndim == 2 and T0.shape[1] == N:
            u = T0.T.copy()
        elif T0.ndim == 2 and T0.shape[0] == N:
            u = T0.copy()
        else:
            raise ValueError(f"T_init shape {T0.shape} does not match N = {N}")
    B = max(u.shape[1], q(0.0).shape[1])
    if u.shape[1] != B:
        u = np.repeat(u, B, axis=1)
    batch = B > 1 or (T_init is not None and np.ndim(T_init) == 2)
    for end in (0, 1):
        i = 0 if end == 0 else N - 1
        if fixed[i]:
            u[i, :] = bc_values[end]

    # Time step (explicit limit from Gershgorin bound of the operator)
    lam_max = np.max(np.abs(diag) + np.r_[0.0, np.abs(lower)] + np.r_[np.abs(upper), 0.0])
    dt_crit = 2.0 / (alpha * lam_max)
    if method == "explicit":
        if dt is None or dt > dt_crit:
            if dt is not None:
                warnings.warn(f"dt = {dt:.3g} exceeds explicit limit {dt_crit:.3g}; using 0.9*limit")
            dt = 0.9 * dt_crit
    elif method in ("implicit", "cn"):
        if dt is None:
            dt = 10.0 * dt_crit
    else:
        raise ValueError(f"Unknown method {method!r}; use explicit, implicit or cn")
    n_steps = int(np.ceil(t_final / float(dt)))
    dt = t_final / n_steps

    # Implicit system: (I - theta*dt*alpha*A) u_new = rhs
    theta = {"explicit": 0.0, "implicit": 1.0, "cn": 0.5}[method]
    if theta > 0:
        s = theta * dt * alpha
        solver = _Thomas(-s * lower, 1.0 - s * diag, -s * upper)
    flux = (alpha * b)[:, None]

    t_saved = [0.0]
    T_saved = [u.copy()] if store_field else None
    p_saved = [u[probe_index].copy()] if probe_index is not None else None
    if probe_index is not None:
        probe_index = int(probe_index)

    q_old = q(0.0)
    for n in range(1, n_steps + 1):
        t = n * dt
        if method == "explicit":
            u = u + dt * (alpha * _matvec(lower, diag, upper, u) + flux + q_old)
            q_old = q(t)
        elif method == "implicit":
            u = solver.solve(u + dt * (flux + q(t)))
        else:
            q_new = q(t)
            rhs = u + 0.5 * dt * alpha * _matvec(lower, diag, upper, u) + dt * (flux + 0.5 * (q_old + q_new))
            u = solver.solve(rhs)
            q_old = q_new

        if n % save_every == 0 or n == n_steps:
            t_saved.append(t)
            if store_field:
                T_saved.append(u.copy())
            if probe_index is not None:
                p_saved.append(u[probe_index].copy())

    out = {"x": x, "t": np.array(t_saved), "dt": dt, "n_steps": n_steps, "method": method}
    if store_field:
        T = np.array(T_saved)                    # (n_saved, N, B)
        out["T"] = T.transpose(0, 2, 1) if batch else T[:, :, 0]
    if probe_index is not None:
        P = np.array(p_saved)                    # (n_saved, B)
        out["probe"] = P if batch else P[:, 0]
    return out


def exact_sine_decay(x, t, alpha, L=1.0):
    """Exact solution for T(x,0) = sin(pi x / L) with T = 0 at both ends."""
    x = np.asarray(x, dtype=float)
    return np.exp(-alpha * (np.pi / L) ** 2 * t) * np.sin(np.pi * x / L)


def _bessel_j0(z):
    """J0 via its power series (fine for the small arguments used in the self-check)."""
    z = np.asarray(z, dtype=float)
    term = np.ones_like(z)
    total = term.copy()
    for m in range(1, 40):
        term = term * (-(z / 2.0) ** 2) / m**2
        total = total + term
    return total


if __name__ == "__main__":
    # 1) Cartesian sine decay
    for method in ("explicit", "implicit", "cn"):
        N = 41
        x = np.linspace(0, 1, N)
        res = solve(1.0, N, 1.0, 0.1, method=method,
                    dt=None if method != "implicit" else (1 / (N - 1)) ** 2,
                    T_init=np.sin(np.pi * x))
        err = np.max(np.abs(res["T"][-1] - exact_sine_decay(x, 0.1, 1.0)))
        print(f"cartesian  {method:9s} max error={err:.3e}")
        assert err < 2e-3, f"{method} error too large"

    # 2) Cylinder, T = 0 at r = 1, T0 = J0(j01 r): exact T = exp(-j01^2 t) J0(j01 r)
    j01 = 2.404825557695773
    N = 41
    r = np.linspace(0, 1, N)
    res = solve(1.0, N, 1.0, 0.1, method="cn", dt=1e-4, geometry="cylindrical",
                bc_types=("neumann", "dirichlet"), T_init=_bessel_j0(j01 * r))
    err = np.max(np.abs(res["T"][-1] - np.exp(-j01**2 * 0.1) * _bessel_j0(j01 * r)))
    print(f"cylinder   cn        max error={err:.3e}")
    assert err < 2e-3

    # 3) Robin cylinder with uniform heating reaches the analytic steady state
    #    T(r) = Tinf + q R/(2 H R) * ... -> T = Tinf + Q/(4 alpha) (R^2 - r^2) + Q R / (2 alpha H)
    Q, H, Tinf, R = 1.0, 2.0, 5.0, 1.0
    res = solve(R, N, 1.0, 20.0, method="implicit", dt=0.05, geometry="cylindrical",
                bc_types=("neumann", "robin"), bc_values=(0, Tinf), bc_coeffs=(0, H),
                T_init=np.full(N, Tinf), source=np.full(N, Q))
    T_ss = Tinf + Q / 4.0 * (R**2 - r**2) + Q * R / (2.0 * H)
    err = np.max(np.abs(res["T"][-1] - T_ss))
    print(f"robin      implicit  steady-state error={err:.3e}")
    assert err < 2e-3

    # 4) Batch of 3 problems equals 3 single runs
    T0s = np.vstack([np.full(N, 20.0), np.full(N, 25.0), np.full(N, 30.0)])
    times = np.array([0.0, 20.0])
    qs = np.array([[0.5, 1.0, 2.0], [0.5, 1.0, 2.0]])
    rb = solve(R, N, 1.0, 2.0, dt=0.01, geometry="cylindrical", bc_types=("neumann", "robin"),
               bc_values=(0, Tinf), bc_coeffs=(0, H), T_init=T0s, source_uniform=qs,
               source_times=times, probe_index=-1, store_field=False)
    for k in range(3):
        rs = solve(R, N, 1.0, 2.0, dt=0.01, geometry="cylindrical", bc_types=("neumann", "robin"),
                   bc_values=(0, Tinf), bc_coeffs=(0, H), T_init=T0s[k], source=np.full(N, qs[0, k]),
                   probe_index=-1)
        assert np.allclose(rb["probe"][:, k], rs["probe"]), "batch mismatch"
    print("batch      matches single runs")
    print("OK")
