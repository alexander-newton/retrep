import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import inv, lstsq
from scipy.optimize import minimize
import time

# ---------- optional HDFE backend ----------
try:
    from pyhdfe import create as _hdfe_create
except Exception:
    _hdfe_create = None  # we'll guard when HDFE is requested


# ========== small utilities ==========
def _ensure_df(X):
    if isinstance(X, np.ndarray):
        return pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])
    return X.copy()

def _as_scalar_or_none(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) == 1:
            return x[0]
    return x

def _indices_from_names_or_idx(df_cols, items):
    if items is None:
        return []
    if not isinstance(items, (list, tuple)):
        items = [items]
    idxs = []
    for it in items:
        if isinstance(it, (int, np.integer)):
            idxs.append(int(it))
        else:
            idxs.append(df_cols.index(str(it)))
    # unique, keep first occurrence order
    seen, out = set(), []
    for i in idxs:
        if i not in seen:
            out.append(i); seen.add(i)
    return out


# ========== PPML-HDFE (no IV) ==========
class PPMLHDFEResult:
    def __init__(self, beta, vcov, mu, colnames, iters, converged, loglik):
        self.beta = beta
        self.vcov = vcov
        self.se   = np.sqrt(np.diag(vcov))
        self.mu   = mu
        self.colnames = colnames  # corresponds to kept columns after collinearity pruning
        self.iters = iters
        self.converged = converged
        self.loglik = loglik

def _build_ids_from_fe_list(fe_list, n_expected):
    """fe_list is a list of arrays/Series (possibly strings). Return (n×G) int ids."""
    cols = []
    for a in fe_list:
        a = np.asarray(a)
        # robust: convert to categorical integer codes
        if a.ndim != 1 or len(a) != n_expected:
            raise ValueError("Each FE vector must be 1-D of length n")
        cols.append(pd.Categorical(a).codes.astype(np.int64))
    ids = np.column_stack(cols) if cols else np.empty((n_expected, 0), dtype=np.int64)
    return ids

def _ppml_hdfe(y, X, fe_list, colnames=None, tol=1e-6, maxiter=200, drop_collinear=True, eps=1e-12):
    """
    Poisson PML with high-dimensional FE via IRLS + pyhdfe absorption.
    Converges on log-likelihood change; robust (Eicker-White) SE.
    """
    if _hdfe_create is None:
        raise ImportError("pyhdfe required for HDFE. Install via `pip install pyhdfe`.")

    y = np.asarray(y).reshape(-1)
    X = np.asarray(X, float)
    n, k = X.shape

    # Build HDFE ids matrix (n×G) with integer codes
    ids = _build_ids_from_fe_list(fe_list, n)
    fe = _hdfe_create(ids, drop_singletons=False)

    init = np.clip(np.mean(y), eps, None)
    W = np.ones(n) * init
    # Initialize
    Xr = fe.residualize(X, weights=W.reshape(-1,1))  # initial residualization
    # Drop collinear columns (after absorption)
    if drop_collinear:
        keep = (Xr.std(axis=0) > 1e-12)
        if not np.all(keep):
            Xr = Xr[:, keep]
            keep_mask = keep
    else:
        keep = np.ones(Xr.shape[1], dtype=bool)
        keep_mask = keep

      # initial weights
    z = (y- init)/init + np.log(init)
    zr = fe.residualize(z.reshape(-1, 1), weights=W.reshape(-1,1)).ravel()


    def loglik(mu):  # Poisson ll up to constant
        return np.mean(y * np.log(np.clip(mu, eps, None)) - mu)

    ll_old = -np.inf
    it = 0
    converged = False
    while it < maxiter:
        start_time = time.time()
        it += 1
        print(f"PPML-HDFE iter {it}, ll={ll_old:.6f}")
        # clamp eta to avoid exp overflow/underflow
        reg = sm.WLS(zr, Xr, weights=W).fit()
        resid = reg.resid
        xb =  np.clip(z - resid, min(np.log(np.min(y)), -6), max(np.log(np.max(y)),16))
        mu = np.exp(xb)

        z = xb + (y - mu)/mu
        zr = fe.residualize(z.reshape(-1, 1), weights=W.reshape(-1,1)).ravel()
        Xr = fe.residualize(X[:, keep_mask], weights=W.reshape(-1,1))
        W = mu

        ll_new = loglik(mu)
        if np.abs(ll_new - ll_old) < tol * (1.0 + np.abs(ll_old)):
            converged = True
            break

        ll_old = ll_new
        time_diff = time.time() - start_time

        if time_diff > 60:
            print("Warning: PPML-HDFE iteration taking too long (>120s). Stopping.")
            break

    # Final res
    res = sm.WLS(zr, Xr, weights=W).fit()
    beta = res.params
    mu = res.fittedvalues
    W = mu

    # Robust VCOV using FE-residualized scores, built on KEPT columns
    sw = np.sqrt(mu)
    Xw = X * sw[:, None]
    Xw_res = fe.residualize(X)[:, keep_mask]
    A = Xw_res.T @ Xw_res
    u = res.resid
    S = Xw_res * u[:, None]
    B = S.T @ S
    Ainv = inv(A)
    vcov = Ainv @ B @ Ainv

    kept_names = (np.array(colnames or [f"x{j}" for j in range(k)])[keep_mask]).tolist()
    beta_kept  = beta

    return PPMLHDFEResult(
        beta=beta_kept, vcov=vcov, mu=mu,
        colnames=kept_names, iters=it,
        converged=converged, loglik=loglik(mu)
    )


# ========== plain IV-PPML (no FE) ==========
class IVPPMLResult:
    def __init__(self, beta, vcov, J, dof, converged, n_iter, colnames):
        self.beta = beta
        self.vcov = vcov
        self.se   = np.sqrt(np.diag(vcov))
        self.J    = J
        self.dof  = dof
        self.converged = converged
        self.n_iter = n_iter
        self.colnames = colnames

def _moments(beta, y, X, Z):
    mu = np.exp(X @ beta)
    g = Z.T @ (y - mu) / y.shape[0]
    return g, mu

def _D(beta, X, Z):
    n = X.shape[0]
    mu = np.exp(X @ beta)
    ZX = Z.T @ (mu[:, None] * X)
    return -ZX / n, mu

def _J(beta, y, X, Z, W):
    g, _ = _moments(beta, y, X, Z)
    return float(g.T @ W @ g)

def _gradJ(beta, y, X, Z, W):
    g, _ = _moments(beta, y, X, Z)
    D, _ = _D(beta, X, Z)
    return (2 * (D.T @ W @ g)).ravel()

def _opt_weight(Z, resid):
    n, m = Z.shape
    M = Z * resid[:, None]
    S = (M.T @ M) / n
    eig = np.linalg.eigvalsh(S)
    jitter = 1e-8 if eig.min() <= 0 else 0.0
    return inv(S + jitter * np.eye(m)), S

def _sandwich(beta, y, X, Z, W, S=None):
    D, mu = _D(beta, X, Z)
    if S is None:
        resid = y - mu
        M = Z * resid[:, None]
        S = (M.T @ M) / X.shape[0]
    A = D.T @ W @ D
    B = D.T @ W @ S @ W @ D
    Ainv = inv(A)
    return Ainv @ B @ Ainv

def iv_ppml(y, X, Z, beta0=None, two_step=True, tol=1e-10, verbose=False, colnames=None):
    y = np.asarray(y).reshape(-1)
    X = np.asarray(X); n, k = X.shape
    Z = np.asarray(Z); m = Z.shape[1]
    if beta0 is None:
        beta0 = np.zeros(k)

    W = inv((Z.T @ Z) / n)
    opt1 = minimize(_J, beta0, args=(y, X, Z, W), jac=_gradJ, method="BFGS",
                    options={"gtol": tol, "disp": verbose, "maxiter": 10000})
    beta = opt1.x
    J1 = _J(beta, y, X, Z, W)

    if two_step:
        _, mu = _moments(beta, y, X, Z)
        resid = y - mu
        Wopt, S = _opt_weight(Z, resid)
        opt2 = minimize(_J, beta, args=(y, X, Z, Wopt), jac=_gradJ, method="BFGS",
                        options={"gtol": tol, "disp": verbose, "maxiter": 10000})
        beta = opt2.x
        J = _J(beta, y, X, Z, Wopt)
        vcov = _sandwich(beta, y, X, Z, Wopt, S=S)
        converged = opt1.success and opt2.success
        nit = (getattr(opt1, "nit", 0) or 0) + (getattr(opt2, "nit", 0) or 0)
        dof = max(m - k, 0)
        return IVPPMLResult(beta, vcov, J, dof, converged, nit, colnames or [f"b{j}" for j in range(k)])
    else:
        vcov = _sandwich(beta, y, X, Z, W)
        converged = opt1.success
        nit = getattr(opt1, "nit", 0) or 0
        dof = max(m - k, 0)
        return IVPPMLResult(beta, vcov, J1, dof, converged, nit, colnames or [f"b{j}" for j in range(k)])


# ========== HDFE helpers (IV case) ==========
class _NullFE:
    def residualize(self, M):
        return M
    def solve(self, v, weights=None):
        return np.zeros_like(v)

def _mk_fe(fe_arrays, n):
    if fe_arrays is None or len(fe_arrays) == 0:
        return _NullFE()
    if _hdfe_create is None:
        raise ImportError("pyhdfe is required for HDFE absorption but is not installed.")
    ids = _build_ids_from_fe_list(fe_arrays, n)
    return _hdfe_create(ids, drop_singletons=False)

def _resid_with_fe(fe, M):
    if M is None:
        return None
    if isinstance(fe, _NullFE):
        return M
    return fe.residualize(M)

def _solve_fe_given_beta(y, offset, fe, tol=1e-10, maxiter=200):
    eta = offset.copy()
    for _ in range(maxiter):
        mu = np.exp(np.clip(eta, -35, 35))
        mu = np.clip(mu, 1e-12, 1e300)
        r  = y - mu
        z  = eta + r / mu
        v = (z - offset)
        W = mu
        r_fe = fe.residualize(v.reshape(-1, 1), weights=W.reshape(-1, 1)).ravel()
        eta_new = z - r_fe
        if np.max(np.abs(eta_new - eta)) < tol:
            eta = eta_new
            break
        eta = eta_new
    mu = np.exp(np.clip(eta, -35, 35))
    mu = np.clip(mu, 1e-12, 1e300)
    return eta, mu

def _iv_ppml_hdfe_gmm(y, X_exog, X_endog, Z_excl, fe_arrays, beta0=None, tol=1e-9, verbose=False):
    """
    IV-PPML with FE absorption inside GMM.
    Uses FE-residualized instruments (unweighted) and profiles FE via Poisson offset at each beta.
    """
    y = y.reshape(-1)
    n = y.shape[0]
    fe = _mk_fe(fe_arrays, n)

    X_exog = np.asarray(X_exog, float)
    X_endog = np.asarray(X_endog, float) if X_endog is not None and X_endog.size else np.zeros((n,0))
    X_est = np.column_stack([X_exog, X_endog])
    k = X_est.shape[1]

    Z_full = X_exog if Z_excl is None or (isinstance(Z_excl, np.ndarray) and Z_excl.size == 0) \
             else np.column_stack([X_exog, np.asarray(Z_excl, float)])
    Z_res = _resid_with_fe(fe, Z_full)
    m = Z_res.shape[1]

    if beta0 is None:
        beta0 = np.zeros(k)

    def compute_mu(beta):
        offset = X_est @ beta
        _, mu = _solve_fe_given_beta(y, offset, fe)
        return mu

    def moments(beta, mu=None):
        if mu is None:
            mu = compute_mu(beta)
        r = y - mu
        g = (Z_res.T @ r) / n
        return g, r, mu

    def D_jac(beta, mu=None):
        if mu is None:
            mu = compute_mu(beta)
        X_res = _resid_with_fe(fe, X_est)
        ZX = (Z_res.T @ (mu[:, None] * X_res)) / n
        return -ZX, mu, X_res

    # step 1
    W1 = inv((Z_res.T @ Z_res) / n)
    def J1(beta): g,_,_ = moments(beta); return float(g.T @ W1 @ g)
    def dJ1(beta): g,_,_ = moments(beta); D,_,_ = D_jac(beta); return (2*(D.T @ W1 @ g)).ravel()
    o1 = minimize(J1, beta0, jac=dJ1, method="BFGS",
                  options={"gtol": tol, "disp": verbose, "maxiter": 5000})
    b1 = o1.x
    g1, r1, mu1 = moments(b1)

    # optimal weight
    M = Z_res * r1[:, None]
    S = (M.T @ M) / n
    eig = np.linalg.eigvalsh(S)
    jitter = 1e-8 if eig.min() <= 0 else 0.0
    Wopt = inv(S + jitter * np.eye(m))

    # step 2
    def J2(beta): g,_,_ = moments(beta); return float(g.T @ Wopt @ g)
    def dJ2(beta): g,_,_ = moments(beta); D,_,_ = D_jac(beta); return (2*(D.T @ Wopt @ g)).ravel()
    o2 = minimize(J2, b1, jac=dJ2, method="BFGS",
                  options={"gtol": tol, "disp": verbose, "maxiter": 5000})
    b = o2.x
    g, r, mu = moments(b)
    D, _, _ = D_jac(b, mu=mu)

    A = D.T @ Wopt @ D
    B = D.T @ Wopt @ S @ Wopt @ D
    Ainv = inv(A)
    vcov = Ainv @ B @ Ainv
    dof = max(m - k, 0)
    converged = o1.success and o2.success
    nit = (getattr(o1, "nit", 0) or 0) + (getattr(o2, "nit", 0) or 0)
    return IVPPMLResult(b, vcov, float(g.T @ Wopt @ g), dof, converged, nit, colnames=None)


# ========== public wrapper ==========
def fit_ppml_or_ivppml(
    y,
    X,
    instruments=None,
    fe_cols=None,          # list of indices/names inside X that are FE ids
    endog_cols=None,       # list of indices/names inside X that are endogenous (NOT FE)
    interest_var=None,     # name or index (or [single]) in original X
    hdfe=True,
    return_meta=True
):
    """
    - No IV:
         * no FE or hdfe=False -> GLM Poisson
         * FE & hdfe=True      -> PPML-HDFE IRLS (fast, no dummy explosion)
    - With IV:
         * no FE or hdfe=False -> IV-PPML GMM
         * FE & hdfe=True      -> IV-PPML GMM with FE absorption
    """
    y_arr = np.asarray(y).reshape(-1)
    X_df = _ensure_df(X)
    orig_cols = list(X_df.columns)

    interest_var = _as_scalar_or_none(interest_var)
    fe_idx    = _indices_from_names_or_idx(orig_cols, fe_cols)
    endog_idx = _indices_from_names_or_idx(orig_cols, endog_cols)
    endog_idx = [j for j in endog_idx if j not in fe_idx]   # FE cannot be endogenous

    # resolve interest (original X position)
    if interest_var is None:
        interest_name = interest_idx_orig = None
    else:
        if isinstance(interest_var, (int, np.integer)):
            interest_idx_orig = int(interest_var)
            interest_name = orig_cols[interest_idx_orig]
        else:
            interest_name = str(interest_var)
            interest_idx_orig = orig_cols.index(interest_name)

    # === NO IV ===
    if instruments is None:
        if fe_idx and hdfe:
            if _hdfe_create is None:
                raise ImportError("pyhdfe required for HDFE. Install via `pip install pyhdfe`.")
            # absorb FE: drop FE columns from X, but pass their ids to HDFE
            fe_names   = [orig_cols[i] for i in fe_idx]
            fe_arrays  = [np.asarray(X_df[nm]) for nm in fe_names]
            X_df2      = X_df.drop(columns=fe_names)
            rem_cols   = list(X_df2.columns)
            X_mat      = X_df2.to_numpy(float)

            res_hdfe   = _ppml_hdfe(y_arr, X_mat, fe_arrays, colnames=rem_cols, tol=1e-8)

            # interest extraction: match by NAME against kept columns
            interest = None
            if interest_name is not None:
                if interest_name in res_hdfe.colnames:
                    j = res_hdfe.colnames.index(interest_name)
                    interest = {"name": interest_name, "index": j,
                                "beta": float(res_hdfe.beta[j]), "se": float(res_hdfe.se[j])}

            meta = {
                "used_model": "PPML_HDFE",
                "X_columns_after_FE_kept": res_hdfe.colnames,
                "iters": res_hdfe.iters,
                "converged": res_hdfe.converged,
                "loglik": res_hdfe.loglik
            }

            if meta["converged"] is False:
                interest['beta'] = np.nan
            return (res_hdfe, meta, interest) if return_meta else (res_hdfe, interest)

        else:
            # plain PPML (no FE absorption)
            X_mat = X_df.to_numpy(float)
            model = sm.GLM(y_arr, X_mat, family=sm.families.Poisson())
            sm_res = model.fit()
            interest = None
            if interest_idx_orig is not None:
                j = interest_idx_orig
                interest = {"name": orig_cols[j], "index": j,
                            "beta": float(sm_res.params[j]), "se": float(sm_res.bse[j])}
            meta = {"used_model": "GLM_PPML", "X_columns": orig_cols}
            return (sm_res, meta, interest) if return_meta else (sm_res, interest)

    # === IV PRESENT ===
    # Drop FE cols from X if any (absorb them in IV-HDFE path)
    if fe_idx:
        fe_names = [orig_cols[i] for i in fe_idx]
        fe_arrays = [np.asarray(X_df[nm]) for nm in fe_names]
        X_df2 = X_df.drop(columns=fe_names)
    else:
        fe_arrays = []
        X_df2 = X_df

    rem_cols = list(X_df2.columns)
    X_mat = X_df2.to_numpy(float)

    # map endog after FE drop
    endog_names = [orig_cols[i] for i in endog_idx if orig_cols[i] in rem_cols]
    endog_after = [rem_cols.index(nm) for nm in endog_names]

    # interest after FE drop
    interest_after = None
    if interest_idx_orig is not None and orig_cols[interest_idx_orig] in rem_cols:
        interest_after = rem_cols.index(orig_cols[interest_idx_orig])

    # split exog/endog
    k_all = X_mat.shape[1]
    exog_idx = [j for j in range(k_all) if j not in endog_after]
    X_exog = X_mat[:, exog_idx]
    X_endog = X_mat[:, endog_after] if endog_after else np.empty((X_mat.shape[0], 0))
    new_X_names = [rem_cols[j] for j in exog_idx] + [rem_cols[j] for j in endog_after]

    # where does interest land in [exog | endog]?
    interest_in_est = None
    if interest_after is not None:
        if interest_after in exog_idx:
            interest_in_est = exog_idx.index(interest_after)
        elif interest_after in endog_after:
            interest_in_est = len(exog_idx) + endog_after.index(interest_after)

    # Build instruments: [exog X | excluded Z]
    if isinstance(instruments, np.ndarray):
        Z_excl = instruments
        Zcols = [f"z{j}" for j in range(instruments.shape[1])]
    else:
        Z_excl = instruments.to_numpy(float)
        Zcols = list(instruments.columns)

    # Route: if NO FE or hdfe=False -> plain IV-PPML (no pyhdfe)
    if (not fe_idx) or (not hdfe):
        X_est = np.column_stack([X_exog, X_endog])
        Z_mat = np.column_stack([X_exog, Z_excl])
        # good PPML start ignoring IV
        beta0 = np.zeros(X_est.shape[1])
        for _ in range(6):
            mu = np.exp(np.clip(X_est @ beta0, -35, 35)).clip(1e-12, None)
            z_work = (X_est @ beta0) + (y_arr - mu) / mu
            XtW = X_est.T * mu
            beta0 = np.linalg.lstsq(XtW @ X_est, XtW @ z_work, rcond=None)[0]

        ivres = iv_ppml(y_arr, X_est, Z_mat, beta0=beta0, two_step=True, colnames=new_X_names)
        interest = None
        if interest_in_est is not None:
            interest = {"name": new_X_names[interest_in_est], "index": interest_in_est,
                        "beta": float(ivres.beta[interest_in_est]), "se": float(ivres.se[interest_in_est])}
        meta = {
            "used_model": "IV_PPML_GMM",
            "X_columns_after_FE_then_[exog|endog]": new_X_names,
            "Z_columns": [f"Xexog:{nm}" for nm in new_X_names[:len(exog_idx)]] + Zcols,
            "interest_idx_in_est": interest_in_est,
        }
        return (ivres, meta, interest) if return_meta else (ivres, interest)

    # Else: FE present AND hdfe=True -> IV-PPML with HDFE absorption
    ivres = _iv_ppml_hdfe_gmm(y_arr, X_exog, X_endog, Z_excl, fe_arrays, beta0=None, tol=1e-9, verbose=False)
    ivres.colnames = new_X_names

    interest = None
    if interest_in_est is not None:
        interest = {"name": new_X_names[interest_in_est], "index": interest_in_est,
                    "beta": float(ivres.beta[interest_in_est]), "se": float(ivres.se[interest_in_est])}
    meta = {
        "used_model": "IV_PPML_HDFE_GMM",
        "X_columns_after_FE_then_[exog|endog]": new_X_names,
        "Z_columns": [f"Xexog:{nm}" for nm in new_X_names[:len(exog_idx)]] + Zcols,
        "interest_idx_in_est": interest_in_est,
    }
    return (ivres, meta, interest) if return_meta else (ivres, interest)




if __name__ == '__main__':
    data = pd.read_stata('./realdata.dta')
    y = data['PowerperWorker']
    X = data[['Form', '_const']]
    fe = [1,2,3]
    res = fit_ppml_or_ivppml(y, X, fe_cols=None, interest_var='Form', hdfe=False, return_meta=True)