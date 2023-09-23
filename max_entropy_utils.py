import numpy as np
from typing import Tuple, List
from scipy.optimize import minimize
from moments_calculations import Moments


def max_entropy_dist(z: np.array,
                     a1: np.array,
                     a2: np.array,
                     a3: np.array,
                     a4: np.array,
                     delta: float, howmuch: float = 20):
    xs = np.arange(a1 - howmuch, a1 + howmuch, delta)
    return - (z[0] + z[1] * a1 + z[2] * a2 + z[3] * a3 + z[4] * a4) + \
        delta * np.sum(np.exp(z[0] + z[1] * xs + z[2] * ((xs - a1)**2) + z[3] * ((xs - a1)**3) + z[4] * ((xs - a1)**4)))


def max_entropy_pdf(z, x, vmu1):
    return np.exp(z[0] + z[1] * x + z[2] * (x - vmu1)**2 + z[3] * (x - vmu1)**3 + z[4]*(x - vmu1)**4)


def calc_max_entropy_dist_params(moments: Moments,
                                 delta: float = 0.01) -> Tuple[List[np.array], np.array, np.array]:

    coeffs0 = np.array([0., 0., 0., 0., -1.])

    n_ev = len(moments.vmu1)
    max_entropy_params = []
    for ev_idx in range(n_ev):
        fit_methods = ['Nelder-Mead', 'Powell']
        # method='trust-constr', options={'maxiter': 500000})
        val_success = False
        for fit_method in fit_methods:
            print(f'EV #{ev_idx + 1}: Trying optimizing the max-entropy function with {fit_method}')
            res = minimize(max_entropy_dist, coeffs0, args=(moments.vmu1[ev_idx],
                                                            moments.vmu2[ev_idx],
                                                            moments.vmu3[ev_idx],
                                                            moments.vmu4[ev_idx], delta),
                        tol=1e-6,
                        method=fit_method,
                        options={'maxiter': 50000})

            if res.success:
            # Sanity Check
                val_success = _validate_fit_converged(moments, ev_idx, res.x, delta)
                if val_success:
                    break
            print(f'EV #{ev_idx + 1}: Fit didn\'t converge: {res.message}')
        if val_success:
            max_entropy_params.append(res.x)
        else:
            max_entropy_params.append(None)

    return (max_entropy_params, moments.vmu1, moments.vmu2)


def _validate_fit_converged(moments: Moments, ev_idx: int, coeffs: List[float], delta: float = 0.01) -> bool:
    xs = np.arange(moments.vmu1[ev_idx] - 3.5 * np.sqrt(moments.vmu2[ev_idx]),
                           moments.vmu1[ev_idx] + 3.5 * np.sqrt(moments.vmu2[ev_idx]), delta)

    unnormed_pdf = max_entropy_pdf(coeffs, xs, moments.vmu1[ev_idx])
    pdf = unnormed_pdf / (delta * sum(unnormed_pdf))

    est_vmu1 = delta * sum(xs * pdf)
    est_vmu2 = delta * sum((xs - est_vmu1)**2 * pdf)

    return (abs(est_vmu1 - moments.vmu1[ev_idx]) / abs(moments.vmu1[ev_idx])) < 0.5 and \
                    (abs(est_vmu2 - moments.vmu2[ev_idx]) / abs(moments.vmu2[ev_idx])) < 0.5
