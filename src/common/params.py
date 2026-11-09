"""params.yaml, and the names derived from it."""

from pathlib import Path

import yaml

PARAMS_PATH = Path("params.yaml")
PLACEHOLDER_STUDENT_ID = "000000000"


def load_params(path: Path = PARAMS_PATH) -> dict:
    """The parameters in params.yaml."""
    return yaml.safe_load(path.read_text())


def model_name(params: dict) -> str:
    """The registered model name, ctr-<student_id>."""
    return f"ctr-{params['student_id']}"


def publisher_column(params: dict) -> str:
    """site_id on site traffic, app_id on app traffic."""
    if params["slice"] not in ("site", "app"):
        raise ValueError(f"params.yaml: slice must be site or app, not {params['slice']!r}")
    return f"{params['slice']}_id"
