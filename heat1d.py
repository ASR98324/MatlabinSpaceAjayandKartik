"""NumPy solver for the 1D heat equation.

    dT/dt = alpha * d2T/dx2 + Q(x, t)        on 0 <= x <= L

Spatial discretization: 2nd-order central differences on a uniform grid.
Time integration:
    'explicit'  Forward Euler (dt limited to h^2 / (2*alpha))
    'implicit'  Backward Euler (unconditionally stable, 1st order in time)
    'cn'        Crank-Nicolson (unconditionally stable, 2nd order in time)
Boundary conditions, one per end (left, right):
    'dirichlet' fixed temperature T = value
    'neumann'   outward normal gradient dT/dn = value (0 = insulated)

Designed to be called from MATLAB (see heat_equation_1d.m), so inputs are
plain arrays rather than Python callables.
"""

import warnings

import numpy as np


def _laplacian(N, h, bc_types, bc_values):
    """Tridiagonal Laplacian (lower, diag, upper), flux vector b, Dirichlet mask.

    Neumann ends use a ghost node T_ghost = T_neighbor + 2*h*g, so the end row
    becomes [-2, 2]/h^2 and 2*g/h goes into b. Dirichlet rows are zeroed so
    those nodes never change.
    """
    lower = np.ones(N - 1)
    diag = -2.0 * np.ones(N)
    upper = np.ones(N - 1)
    b = np.zeros(N)
    fixed = np.zeros(N, dtype=bool)

    for end, (kind, value) in enumerate(zip(bc_types, bc_values)):
        i = 0 if end == 0 else N - 1
        kind = str(kind).lower()
        if kind == "neumann":
            if end == 0:
                upper[0] = 2.0
            else:
                lower[-1] = 2.0
            b[i] = 2.0 * value / h
        elif kind == "dirichlet":
            fixed[i] = True
        else:
            raise ValueError(f"Unknown BC type {kind!r}")

    lower, diag, upper = lower / h**2, diag / h**2, upper / h**2
    # Zero Dirichlet rows: row i owns diag[i], lower[i-1], upper[i]
    diag[fixed] = 0.0
    b[fixed] = 0.0
    if fixed[0]:
        upper[0] = 0.0
    if fixed[-1]:
        lower[-1] = 0.0
    return lower, diag, upper, b, fixed


def _matvec(lower, diag, upper, u):
    r = diag * u
    r[1:] += lower * u[:-1]
    r[:-1] += upper * u[1:]
    return r


class _Thomas:
    """Tridiagonal solver; factorization is done once and reused every step."""

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
        N = d.size
        lower, cp, denom = self.lower, self.cp, self.denom
        dp = np.empty(N)
        dp[0] = d[0] / denom[0]
        for i in range(1, N):
            dp[i] = (d[i] - lower[i - 1] * dp[i - 1]) / denom[i]
        x = np.empty(N)
        x[-1] = dp[-1]
        for i in range(N - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x


def _source_fn(source, source_times, N, keep):
    """Turn None / length-N array / (n_times x N) table into q(t)."""
    if source is None:
        zero = np.zeros(N)
        return lambda t: zero

    src = np.asarray(source, dtype=float)
    if src.size == N:
        q = src.ravel() * keep
        return lambda t: q

    times = np.asarray(source_times, dtype=float).ravel()
    nt = times.size
    if src.shape != (nt, N):
        if src.shape == (N, nt):
            src = src.T
        else:
            src = src.reshape(nt, N)
    src = src * keep

    def q(t):
        if t <= times[0]:
            return src[0]
        if t >= times[-1]:
            return src[-1]
        k = np.searchsorted(times, t)
        w = (t - times[k - 1]) / (times[k] - times[k - 1])
        return (1.0 - w) * src[k - 1] + w * src[k]

    return q


def solve(L, N, alpha, t_final, method="cn", dt=None,
          bc_types=("dirichlet", "dirichlet"), bc_values=(0.0, 0.0),
          T_init=None, source=None, source_times=None, save_every=1):
    """Integrate the 1D heat equation.

    Parameters
    ----------
    L, N, alpha, t_final : rod length, number of grid nodes, diffusivity, end time
    method      : 'explicit', 'implicit' or 'cn'
    dt          : time step (None -> automatic)
    bc_types    : (left, right), each 'dirichlet' or 'neumann'
    bc_values   : (left, right) temperature or outward gradient
    T_init      : length-N initial temperature (default zeros)
    source      : None, length-N array (constant Q), or (n_times x N) table
    source_times: times for the rows of a source table (linearly interpolated)
    save_every  : store a snapshot every this many steps

    Returns
    -------
    dict with 'x' (N,), 't' (n_saved,), 'T' (n_saved, N), 'dt', 'n_steps', 'method'
    """
    N = int(N)
    L, alpha, t_final = float(L), float(alpha), float(t_final)
    method = str(method).lower()
    bc_types = [str(s) for s in bc_types]
    bc_values = [float(v) for v in np.asarray(bc_values, dtype=float).ravel()]
    save_every = max(int(save_every), 1)

    x = np.linspace(0.0, L, N)
    h = x[1] - x[0]
    lower, diag, upper, b, fixed = _laplacian(N, h, bc_types, bc_values)
    keep = (~fixed).astype(float)
    q = _source_fn(source, source_times, N, keep)

    u = np.zeros(N) if T_init is None else np.asarray(T_init, dtype=float).ravel().copy()
    if u.size != N:
        raise ValueError(f"T_init has {u.size} values, expected {N}")
    for end in (0, 1):
        if fixed[0 if end == 0 else -1]:
            u[0 if end == 0 else -1] = bc_values[end]

    # Time step
    dt_crit = h**2 / (2.0 * alpha)
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
    flux = alpha * b

    t_saved = [0.0]
    T_saved = [u.copy()]
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
            T_saved.append(u.copy())

    return {
        "x": x,
        "t": np.array(t_saved),
        "T": np.array(T_saved),
        "dt": dt,
        "n_steps": n_steps,
        "method": method,
    }


def exact_sine_decay(x, t, alpha, L=1.0):
    """Exact solution for T(x,0) = sin(pi x / L) with T = 0 at both ends."""
    x = np.asarray(x, dtype=float)
    return np.exp(-alpha * (np.pi / L) ** 2 * t) * np.sin(np.pi * x / L)


if __name__ == "__main__":
    # Self-check against the exact solution
    for method in ("explicit", "implicit", "cn"):
        N = 41
        x = np.linspace(0, 1, N)
        res = solve(1.0, N, 1.0, 0.1, method=method,
                    dt=None if method != "implicit" else (1 / (N - 1)) ** 2,
                    T_init=np.sin(np.pi * x))
        err = np.max(np.abs(res["T"][-1] - exact_sine_decay(x, 0.1, 1.0)))
        print(f"{method:9s} steps={res['n_steps']:5d}  max error={err:.3e}")
        assert err < 2e-3, f"{method} error too large"
    print("OK")
