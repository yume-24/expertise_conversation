"""REML models and ML parametric-bootstrap tests preserved from notebook 02."""
import warnings
import numpy as np
import pandas as pd
from scipy import optimize
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from .plots import METRIC_LABELS as METRICS

LEVELS = ['child', 'teenager', 'undergraduate', 'graduate', 'expert']
REPORT_COLUMNS = ['label', 'adjustment', 'n_conversations', 'n_videos', 'beta_raw', 'ci_low_raw', 'ci_high_raw',
                  'expert_minus_child_raw', 'p_bootstrap', 'p_bootstrap_mcse', 'p_fdr_bh_8', 'p_holm_8']


def prepare_data(raw):
    required = ["dataset", "conversation_id", "video_id", "level", "level_label", "n_utterances", *METRICS]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing source columns: {missing}")
    if raw.duplicated(["dataset", "conversation_id"]).any():
        raise ValueError("Duplicate conversation IDs in source.")
    for col in ["level", "n_utterances", *METRICS]:
        raw[col] = pd.to_numeric(raw[col], errors="raise")
    valid = raw[required].notna().all(axis=1)
    valid &= np.isfinite(raw[["level", "n_utterances", *METRICS]]).all(axis=1)
    valid &= raw["n_utterances"].gt(1)
    excluded = raw.loc[~valid, required].copy()
    data = raw.loc[valid, required].copy()
    data["video_group"] = data["dataset"].astype(str) + "::" + data["video_id"].astype(str)
    if not data["level"].isin(range(5)).all():
        raise ValueError("Expected expertise levels 0 through 4.")
    if not data["level_label"].eq(data["level"].map(dict(enumerate(LEVELS)))).all():
        raise ValueError("Expertise labels disagree with scores.")
    if data.duplicated(["video_group", "level"]).any():
        raise ValueError("Expected at most one conversation per video and expertise level.")
    data["expertise_c"] = data["level"] - 2.0
    data["log_n_c"] = np.log(data["n_utterances"]) - np.log(data["n_utterances"]).mean()
    data = data.sort_values(["video_group", "level"]).reset_index(drop=True)
    if data["video_group"].nunique() < 5 or len(data) < 15:
        raise ValueError("Too few observations/groups for this analysis plan.")
    for metric in METRICS:
        if data[metric].std(ddof=1) <= 0:
            raise ValueError(f"Constant outcome: {metric}")
    coverage = pd.crosstab(data["video_group"], data["level_label"]).reindex(columns=LEVELS, fill_value=0)

    if not coverage.eq(1).all().all():
        warnings.warn("Incomplete matched video sets: inspect coverage and exclusions.")
    return data, excluded, coverage


def fit_reml(frame, formula):
    candidates = []
    messages = []
    for method in ("lbfgs", "powell"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = smf.mixedlm(formula, frame, groups=frame["video_group"], re_formula="1").fit(
                    reml=True, method=method, maxiter=1500, disp=False)
                if result.converged and np.isfinite(result.llf) and np.isfinite(result.bse_fe).all():
                    candidates.append(result)
            except (ValueError, np.linalg.LinAlgError) as exc:
                messages.append(f"{method}: {exc}")
            messages.extend(f"{method}: {w.message}" for w in caught)
    if not candidates:
        raise RuntimeError(f"No usable converged REML fit for {formula}: {messages}")
    result = max(candidates, key=lambda fit: fit.llf)
    return result, " | ".join(dict.fromkeys(messages))


def fit_models(data):
    fits, frames, records = {}, {}, []
    for metric, label in METRICS.items():
        frame = data.copy()
        outcome_sd = frame[metric].std(ddof=1)
        frame["outcome_z"] = (frame[metric] - frame[metric].mean()) / outcome_sd
        frames[metric] = frame
        for adjustment, formula in {
            "Unadjusted": "outcome_z ~ expertise_c",
            "Count-adjusted": "outcome_z ~ expertise_c + log_n_c",
        }.items():
            result, fit_warnings = fit_reml(frame, formula)
            fits[(metric, adjustment)] = result
            beta = result.fe_params["expertise_c"]
            lo, hi = result.conf_int().loc["expertise_c"]
            video_var = float(np.asarray(result.cov_re)[0, 0])
            records.append(dict(metric=metric, label=label, adjustment=adjustment,
                n_conversations=len(frame), n_videos=frame.video_group.nunique(),
                beta_sd=beta, se_sd=result.bse_fe["expertise_c"], ci_low_sd=lo, ci_high_sd=hi,
                beta_raw=beta*outcome_sd, ci_low_raw=lo*outcome_sd, ci_high_raw=hi*outcome_sd,
                expert_minus_child_raw=4*beta*outcome_sd,
                expert_minus_child_low_raw=4*lo*outcome_sd, expert_minus_child_high_raw=4*hi*outcome_sd,
                p_wald_asymptotic=result.pvalues["expertise_c"],
                log_count_beta_sd=result.fe_params.get("log_n_c", np.nan),
                video_variance=video_var, residual_variance=result.scale,
                icc=video_var/(video_var+result.scale),
                boundary_variance=video_var < 1e-6*result.scale,
                converged=result.converged, fit_warnings=fit_warnings))
    results = pd.DataFrame(records)

    return results, fits, frames


def profile_ml(y, X, Z, group_sizes):
    """Exact profiled ML for a single Gaussian random intercept; allows tau² = 0."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    n = len(y)
    if np.linalg.matrix_rank(X) != X.shape[1]:
        raise ValueError("Rank-deficient fixed-effect design.")
    gx, gy = Z.T @ X, Z.T @ y
    xx, xy, yy = X.T @ X, X.T @ y, y @ y

    def evaluate(r, details=False):
        weights = r / (1 + r * group_sizes)
        xtvi_x = xx - gx.T @ (weights[:, None] * gx)
        xtvi_y = xy - gx.T @ (weights * gy)
        beta = np.linalg.solve(xtvi_x, xtvi_y)
        rss = yy - gy @ (weights * gy) - xtvi_y @ beta
        if rss <= 0 or not np.isfinite(rss):
            raise FloatingPointError("Nonpositive profiled residual sum of squares.")
        sigma2 = rss / n
        nll = 0.5 * (n * (np.log(2*np.pi) + 1 + np.log(sigma2))
                     + np.log1p(r*group_sizes).sum())
        if details:
            return dict(llf=-nll, beta=beta, sigma2=sigma2, tau2=r*sigma2, ratio=r)
        return nll

    # Optimize log(1+r) for stable coverage of small and large variance ratios.
    upper = np.log1p(1e6)
    opt = optimize.minimize_scalar(lambda t: evaluate(np.expm1(t)),
                                  bounds=(0, upper), method="bounded",
                                  options={"xatol": 1e-8})
    if not opt.success or opt.x > upper - 0.01:
        raise RuntimeError("Variance optimization failed or reached upper search bound.")
    ratio = float(np.expm1(opt.x))
    if evaluate(0.0) <= opt.fun:
        ratio = 0.0
    return evaluate(ratio, details=True)


def checked_lr(null, full):
    lr = 2 * (full["llf"] - null["llf"])
    if lr < -1e-6:
        raise RuntimeError("Nested full-model likelihood is below null likelihood.")
    return max(0.0, lr)


def validate_ml(y, X, reference, group_codes):
    candidates = []
    for method in ("lbfgs", "powell"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import statsmodels.api as sm
                result = sm.MixedLM(y, X, groups=group_codes).fit(reml=False, method=method, disp=False)
                if result.converged and np.isfinite(result.llf):
                    candidates.append(result)
            except (ValueError, np.linalg.LinAlgError):
                pass
    if not candidates:
        raise RuntimeError("No converged statsmodels ML validation fit.")
    best = max(candidates, key=lambda r: r.llf)
    np.testing.assert_allclose(reference["llf"], best.llf, atol=2e-4, rtol=0)
    np.testing.assert_allclose(reference["beta"], best.fe_params, atol=2e-4, rtol=0)
    return abs(reference["llf"] - best.llf)


def bootstrap_tests(data, results, frames, seed, n_bootstrap, alpha):
    SEED, N_BOOTSTRAP, ALPHA = seed, n_bootstrap, alpha
    group_codes, group_names = pd.factorize(data["video_group"], sort=True)
    group_sizes = np.bincount(group_codes)
    Z = np.eye(len(group_sizes))[group_codes]

    bootstrap_records, bootstrap_draws = [], []
    seed_sequences = np.random.SeedSequence(SEED).spawn(len(results))
    for index, row in results.iterrows():
        frame = frames[row.metric]
        y = frame["outcome_z"].to_numpy()
        X0 = np.ones((len(frame), 1))
        if row.adjustment == "Count-adjusted":
            X0 = np.column_stack([X0, frame["log_n_c"]])
        X1 = np.column_stack([X0, frame["expertise_c"]])
        null, full = profile_ml(y, X0, Z, group_sizes), profile_ml(y, X1, Z, group_sizes)
        validation_error = max(validate_ml(y, X0, null, group_codes), validate_ml(y, X1, full, group_codes))
        observed = checked_lr(null, full)
        rng = np.random.default_rng(seed_sequences[index])
        simulated_lr = np.empty(N_BOOTSTRAP)
        for b in range(N_BOOTSTRAP):
            video_effects = rng.normal(0, np.sqrt(null["tau2"]), len(group_sizes))
            simulated_y = X0 @ null["beta"] + video_effects[group_codes] + rng.normal(0, np.sqrt(null["sigma2"]), len(y))
            simulated_lr[b] = checked_lr(profile_ml(simulated_y, X0, Z, group_sizes), profile_ml(simulated_y, X1, Z, group_sizes))
        exceedances = int(np.count_nonzero(simulated_lr >= observed))
        p_boot = (1 + exceedances) / (N_BOOTSTRAP + 1)
        bootstrap_records.append(dict(metric=row.metric, adjustment=row.adjustment,
            lr_observed=observed, p_bootstrap=p_boot,
            p_bootstrap_mcse=np.sqrt(p_boot*(1-p_boot)/N_BOOTSTRAP),
            bootstrap_replicates=N_BOOTSTRAP, bootstrap_exceedances=exceedances,
            ml_validation_max_error=validation_error))
        bootstrap_draws.append(pd.DataFrame({"metric": row.metric, "adjustment": row.adjustment,
                                            "replicate": np.arange(N_BOOTSTRAP), "null_lr": simulated_lr}))
        print(f"{row.label}, {row.adjustment}: bootstrap p={p_boot:.4f}", flush=True)
    results = results.merge(pd.DataFrame(bootstrap_records), on=["metric", "adjustment"], validate="one_to_one")
    results['p_fdr_bh_8'] = multipletests(results['p_bootstrap'], method='fdr_bh')[1]
    results['p_holm_8'] = multipletests(results['p_bootstrap'], method='holm')[1]
    results['passes_fdr_05'] = results['p_fdr_bh_8'] < ALPHA
    return results, pd.concat(bootstrap_draws, ignore_index=True)


def diagnostic_data(data, results, fits, frames):
    group_codes, _ = pd.factorize(data['video_group'], sort=True)
    group_sizes = np.bincount(group_codes)
    Z = np.eye(len(group_sizes))[group_codes]
    records = []
    for row in results.itertuples():
        fit = fits[(row.metric, row.adjustment)]
        frame = frames[row.metric]
        fixed_prediction = np.asarray(fit.model.exog @ fit.fe_params)
        marginal_residual = frame['outcome_z'].to_numpy() - fixed_prediction
        tau2 = float(np.asarray(fit.cov_re)[0, 0])
        offsets = (tau2 / (fit.scale + group_sizes*tau2)) * (Z.T @ marginal_residual)
        fitted = fixed_prediction + offsets[group_codes]
        residual = frame['outcome_z'].to_numpy() - fitted
        records.append(frame[['dataset', 'conversation_id']].assign(metric=row.metric,
                       adjustment=row.adjustment, fitted=fitted, residual=residual))
    return pd.concat(records, ignore_index=True)
