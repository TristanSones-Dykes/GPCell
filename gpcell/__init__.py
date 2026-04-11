from .backend import TimeVaryingCosine
from .functional import (
    BICResult,
    BootstrapResult,
    DetectorConfig,
    DetectorData,
    PipelineResult,
    PreparedData,
    fit_bic,
    fit_bootstrap,
    load_detector_data,
    prepare_detector_data,
    run_pipeline,
)
from .main import OscillatorDetector

__all__ = [
    "OscillatorDetector",
    "TimeVaryingCosine",
    "DetectorConfig",
    "DetectorData",
    "PreparedData",
    "BICResult",
    "BootstrapResult",
    "PipelineResult",
    "load_detector_data",
    "prepare_detector_data",
    "fit_bic",
    "fit_bootstrap",
    "run_pipeline",
]
