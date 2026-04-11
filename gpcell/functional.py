from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Sequence

import numpy as np
from gpflow.kernels import Cosine, Kernel, Matern12, White
from numpy.random import multivariate_normal, uniform
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

from gpcell.backend import (
    GPPriorFactory,
    GPPriorTrainingFlag,
    GaussianProcess,
    hes_ou_prior,
    hes_ouosc_prior,
    ou_trainables,
    ouosc_trainables,
)
from gpcell.utils import background_noise, detrend, fit_processes, fit_processes_joblib, load_data


@dataclass(frozen=True)
class DetectorConfig:
    verbose: bool = False
    set_noise: float | None = None
    joblib: bool = False
    detrend_enabled: bool = True
    ou_prior_gen: GPPriorFactory = hes_ou_prior
    ouosc_prior_gen: GPPriorFactory = hes_ouosc_prior
    ou_trainables: GPPriorTrainingFlag = field(default_factory=lambda: ou_trainables)
    ouosc_trainables: GPPriorTrainingFlag = field(default_factory=lambda: ouosc_trainables)
    bic_replicates: int = 10
    bootstrap_replicates: int = 10


@dataclass(frozen=True)
class DetectorData:
    X: Sequence[np.ndarray]
    Y: Sequence[np.ndarray]
    X_bckgd: Sequence[np.ndarray] | None = None
    bckgd: Sequence[np.ndarray] | None = None

    @property
    def n_cells(self) -> int:
        return len(self.Y)


@dataclass(frozen=True)
class PreparedData:
    X: Sequence[np.ndarray]
    Y: Sequence[np.ndarray]
    Y_detrended: Sequence[np.ndarray]
    noise_list: list[float]
    mean_noise: float | None
    bckgd_GPs: list[GaussianProcess] | None
    detrend_GPs: list[GaussianProcess] | None


@dataclass(frozen=True)
class BICResult:
    LLRs: list[float]
    BIC_diffs: list[float]
    k_ou_list: list[Kernel]
    k_ouosc_list: list[Kernel]
    periods: list[float]
    ou_GPs: list[list[GaussianProcess]]
    ouosc_GPs: list[list[GaussianProcess]]


@dataclass(frozen=True)
class BootstrapResult:
    synth_LLRs: list[float]
    osc_filt: np.ndarray


@dataclass(frozen=True)
class PipelineResult:
    prepared: PreparedData
    bic: BICResult | None = None
    bootstrap: BootstrapResult | None = None


def load_detector_data(
    path: str, x_name: str, background_name: str, y_name: str, *, set_noise: float | None = None
) -> DetectorData:
    X, Y = load_data(path, x_name, y_name)
    if set_noise is not None:
        return DetectorData(X=X, Y=Y)
    X_bckgd, bckgd = load_data(path, x_name, background_name)
    return DetectorData(X=X, Y=Y, X_bckgd=X_bckgd, bckgd=bckgd)


def prepare_detector_data(data: DetectorData, config: DetectorConfig) -> PreparedData:
    if config.set_noise is None:
        if data.X_bckgd is None or data.bckgd is None:
            raise ValueError("Background traces are required unless set_noise is provided.")
        mean_noise, bckgd_gps = background_noise(
            data.X_bckgd, data.bckgd, 7.0, verbose=config.verbose
        )
        noise_list = [float(mean_noise / np.std(y)) for y in data.Y]
    else:
        mean_noise, bckgd_gps = None, None
        noise_list = [float(config.set_noise) for _ in data.Y]

    if config.detrend_enabled:
        y_detrended, detrend_gps = detrend(data.X, data.Y, 7.0, verbose=config.verbose)
    else:
        y_detrended, detrend_gps = data.Y, None

    return PreparedData(
        X=data.X,
        Y=data.Y,
        Y_detrended=y_detrended,
        noise_list=noise_list,
        mean_noise=float(mean_noise) if mean_noise is not None else None,
        bckgd_GPs=bckgd_gps,
        detrend_GPs=detrend_gps,
    )


def _build_prior_lists(config: DetectorConfig, noise_list: Sequence[float]):
    if config.joblib:
        ou_prior = config.ou_prior_gen
        ouosc_prior = config.ouosc_prior_gen
        if callable(ou_prior):
            ou_priors = [partial(ou_prior, noise) for noise in noise_list]
        else:
            ou_priors = [partial(prior, noise) for prior, noise in zip(ou_prior, noise_list)]
        if callable(ouosc_prior):
            ouosc_priors = [partial(ouosc_prior, noise) for noise in noise_list]
        else:
            ouosc_priors = [partial(prior, noise) for prior, noise in zip(ouosc_prior, noise_list)]
        return ou_priors, ouosc_priors

    ou_priors = [
        (
            lambda noise=noise: {
                "kernel.lengthscales": uniform(0.1, 2.0),
                "kernel.variance": uniform(0.1, 2.0),
                "likelihood.variance": noise**2,
            }
        )
        for noise in noise_list
    ]
    ouosc_priors = [
        (
            lambda noise=noise: {
                "kernel.kernels[0].lengthscales": uniform(0.1, 2.0),
                "kernel.kernels[0].variance": uniform(0.1, 2.0),
                "kernel.kernels[1].lengthscales": uniform(0.1, 4.0),
                "likelihood.variance": noise**2,
            }
        )
        for noise in noise_list
    ]
    return ou_priors, ouosc_priors


def _fit_ou_ouosc(
    X: Sequence[np.ndarray],
    Y: Sequence[np.ndarray],
    ou_priors,
    ou_trainables: GPPriorTrainingFlag,
    ouosc_priors,
    ouosc_trainables: GPPriorTrainingFlag,
    replicates: int,
    joblib: bool,
    verbose: bool,
    mcmc: bool = False,
):
    def normalize_fit_groups(groups, n_traces: int):
        groups = list(groups)
        if len(groups) == 1 and len(groups[0]) == n_traces:
            return [[process] for process in groups[0]]
        return groups

    if joblib:
        fit_fn = fit_processes_joblib
    else:
        fit_fn = fit_processes
    ou_gps = fit_fn(
        X,
        Y,
        Matern12,
        ou_priors,
        replicates=replicates,
        trainable=ou_trainables,
        mcmc=mcmc,
        verbose=verbose,
    )
    ouosc_gps = fit_fn(
        X,
        Y,
        [Matern12, Cosine],
        ouosc_priors,
        replicates=replicates,
        trainable=ouosc_trainables,
        mcmc=mcmc,
        verbose=verbose,
    )
    return normalize_fit_groups(ou_gps, len(Y)), normalize_fit_groups(ouosc_gps, len(Y))


def _calc_gpr_bic_llr(
    X: Sequence[np.ndarray],
    Y: Sequence[np.ndarray],
    ou_GPs: list[list[GaussianProcess]],
    ouosc_GPs: list[list[GaussianProcess]],
    *,
    joblib: bool,
    noise_list: Sequence[float],
    ou_prior_gen,
    ou_trainables_val: GPPriorTrainingFlag,
    ouosc_prior_gen,
    ouosc_trainables_val: GPPriorTrainingFlag,
):
    LLRs = []
    BIC_diffs = []
    k_max_ou_list = []
    k_max_ouosc_list = []
    periods = []

    fit_fn = fit_processes_joblib if joblib else fit_processes

    for i, (x, y, ou, ouosc) in enumerate(zip(X, Y, ou_GPs, ouosc_GPs)):
        ou_LL = [gp.log_posterior() for gp in ou]
        ouosc_LL = [gp.log_posterior() for gp in ouosc]

        ou_nan, ouosc_nan = np.isnan(ou_LL), np.isnan(ouosc_LL)
        if ou_nan.any() or ouosc_nan.any():
            ou_all, ouosc_all = ou_nan.all(), ouosc_nan.all()
            attempts = 0
            while ou_all or ouosc_all:
                attempts += 1
                if ou_all:
                    ou = next(
                        iter(
                            fit_fn(
                                [x],
                                [y],
                                Matern12,
                                [lambda noise=noise_list[i]: ou_prior_gen(noise)],
                                replicates=10,
                                trainable=ou_trainables_val,
                            )
                        )
                    )
                if ouosc_all:
                    ouosc = next(
                        iter(
                            fit_fn(
                                [x],
                                [y],
                                [Matern12, Cosine],
                                [lambda noise=noise_list[i]: ouosc_prior_gen(noise)],
                                replicates=10,
                                trainable=ouosc_trainables_val,
                            )
                        )
                    )
                if ou_all:
                    ou_LL = [gp.log_posterior() for gp in ou]
                    ou_nan = np.isnan(ou_LL)
                if ouosc_all:
                    ouosc_LL = [gp.log_posterior() for gp in ouosc]
                    ouosc_nan = np.isnan(ouosc_LL)
                ou_all, ouosc_all = ou_nan.all(), ouosc_nan.all()
                if (ou_all or ouosc_all) and attempts >= 5:
                    ou_LL = np.zeros(10)
                    ou_nan = np.isnan(ou_LL)
                    ouosc_LL = np.zeros(10)
                    ouosc_nan = np.isnan(ouosc_LL)
                    break

        ou_LL = np.array(ou_LL)[~ou_nan]
        ouosc_LL = np.array(ouosc_LL)[~ouosc_nan]

        max_ou_ll = np.max(ou_LL)
        max_ouosc_ll = np.max(ouosc_LL)
        k_max_ou = ou[int(np.argmax(ou_LL))].fit_gp.kernel
        k_max_ouosc = ouosc[int(np.argmax(ouosc_LL))].fit_gp.kernel
        k_ouosc = ouosc[0].fit_gp.kernel

        llr = float(100 * 2 * (max_ouosc_ll - max_ou_ll) / len(y))
        bic_ouosc = float(-2 * max_ouosc_ll + 3 * np.log(len(y)))
        bic_ou = float(-2 * max_ou_ll + 2 * np.log(len(y)))
        bic_diff = float(bic_ou - bic_ouosc)

        cov_ou_osc = k_ouosc(x).numpy()[0, :]
        peaks, _ = find_peaks(cov_ou_osc, height=0)
        period = float(x[peaks[0]].item()) if len(peaks) else 0.0

        LLRs.append(llr)
        BIC_diffs.append(bic_diff)
        k_max_ou_list.append(k_max_ou)
        k_max_ouosc_list.append(k_max_ouosc)
        periods.append(period)

    return LLRs, BIC_diffs, k_max_ou_list, k_max_ouosc_list, periods


def fit_bic(prepared: PreparedData, config: DetectorConfig) -> BICResult:
    ou_priors, ouosc_priors = _build_prior_lists(config, prepared.noise_list)
    ou_gps, ouosc_gps = _fit_ou_ouosc(
        prepared.X,
        prepared.Y_detrended,
        ou_priors,
        config.ou_trainables,
        ouosc_priors,
        config.ouosc_trainables,
        replicates=config.bic_replicates,
        joblib=config.joblib,
        verbose=config.verbose,
    )
    llrs, bic_diffs, k_ou, k_ouosc, periods = _calc_gpr_bic_llr(
        prepared.X,
        prepared.Y_detrended,
        ou_gps,
        ouosc_gps,
        joblib=config.joblib,
        noise_list=prepared.noise_list,
        ou_prior_gen=config.ou_prior_gen,
        ou_trainables_val=config.ou_trainables,
        ouosc_prior_gen=config.ouosc_prior_gen,
        ouosc_trainables_val=config.ouosc_trainables,
    )
    return BICResult(
        LLRs=llrs,
        BIC_diffs=bic_diffs,
        k_ou_list=k_ou,
        k_ouosc_list=k_ouosc,
        periods=periods,
        ou_GPs=ou_gps,
        ouosc_GPs=ouosc_gps,
    )


def fit_bootstrap(
    prepared: PreparedData, config: DetectorConfig, bic: BICResult
) -> BootstrapResult:
    if prepared.detrend_GPs is None:
        raise ValueError("Bootstrap requires detrended traces and detrend GP models.")
    ou_priors, ouosc_priors = _build_prior_lists(config, prepared.noise_list)

    synth_llrs: list[float] = []
    for i in range(len(prepared.Y)):
        x = prepared.X[i]
        noise = prepared.noise_list[i]
        k_se = prepared.detrend_GPs[i].fit_gp.kernel
        k_ou = bic.k_ou_list[i]
        k_white = White(variance=noise**2)  # type: ignore[arg-type]
        k_synth = k_se + k_ou + k_white
        synths = [
            multivariate_normal(np.zeros(len(x)), k_synth(x)).reshape(-1, 1)
            for _ in range(config.bootstrap_replicates)
        ]  # type: ignore[misc]
        synths_detrended, _ = detrend([x for _ in range(config.bootstrap_replicates)], synths, 7.0)
        ou_gps, ouosc_gps = _fit_ou_ouosc(
            [x for _ in range(config.bootstrap_replicates)],
            synths_detrended,
            ou_priors[i],
            config.ou_trainables,
            ouosc_priors[i],
            config.ouosc_trainables,
            replicates=config.bootstrap_replicates,
            joblib=config.joblib,
            verbose=config.verbose,
        )
        llrs, _, _, _, _ = _calc_gpr_bic_llr(
            [x for _ in range(config.bootstrap_replicates)],
            synths_detrended,
            ou_gps,
            ouosc_gps,
            joblib=config.joblib,
            noise_list=[noise for _ in range(config.bootstrap_replicates)],
            ou_prior_gen=config.ou_prior_gen,
            ou_trainables_val=config.ou_trainables,
            ouosc_prior_gen=config.ouosc_prior_gen,
            ouosc_trainables_val=config.ouosc_trainables,
        )
        synth_llrs.extend(llrs)

    llr_array = np.array(bic.LLRs)
    synth_array = np.array(synth_llrs)
    nan_inds = np.isnan(llr_array)
    llr_array = llr_array[~nan_inds]
    synth_array = synth_array[~np.isnan(synth_array)]
    llr_array[llr_array < 0] = 0
    synth_array[synth_array < 0] = 0
    llr_combined = np.concatenate((llr_array, synth_array), 0)

    upper = np.max(llr_combined)
    lower1 = np.min(llr_combined)
    lower = upper - 0.9 * (upper - lower1)
    grid = np.linspace(lower, upper, 20)
    piest = np.zeros_like(grid)
    for i, cutoff in enumerate(grid):
        piest[i] = (np.sum(llr_array < cutoff) / len(llr_array)) / (
            np.sum(synth_array < cutoff) / len(synth_array)
        )
    xx = np.linspace(lower, upper, 100)
    cs = CubicSpline(grid, piest)
    pi_guess = cs(xx)[0]

    llr_idx = np.argsort(llr_array)
    llr_sorted = llr_array[llr_idx]
    q1 = np.zeros_like(llr_sorted)
    for i, thresh in enumerate(llr_sorted):
        q1[i] = (
            pi_guess
            * (np.sum(synth_array >= thresh) / len(synth_array))
            / (np.sum(llr_sorted >= thresh) / len(llr_sorted))
        )
    q_vals = q1[np.argsort(llr_idx)]
    return BootstrapResult(synth_LLRs=synth_llrs, osc_filt=q_vals < 0.05)


def run_pipeline(
    data: DetectorData, config: DetectorConfig, methods: str | Sequence[str] | None = None
) -> PipelineResult:
    prepared = prepare_detector_data(data, config)
    if methods is None:
        methods = ["BIC"]
    elif isinstance(methods, str):
        methods = [methods]
    methods = [m.upper() for m in methods]
    bic_result: BICResult | None = None
    bootstrap_result: BootstrapResult | None = None
    for method in methods:
        if method == "BIC":
            if bic_result is None:
                bic_result = fit_bic(prepared, config)
        elif method == "BOOTSTRAP":
            if bic_result is None:
                bic_result = fit_bic(prepared, config)
            bootstrap_result = fit_bootstrap(prepared, config, bic_result)
        elif method == "MCMC":
            # Kept on class adapter path for now.
            continue
        else:
            raise ValueError(f"Unknown fitting method: {method}")
    return PipelineResult(prepared=prepared, bic=bic_result, bootstrap=bootstrap_result)
