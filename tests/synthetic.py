"""The synthetic drop the public tests run on: shared by conftest.py and the tests, which cannot import conftest."""

import datetime
from pathlib import Path

from src.common.params import PARAMS_PATH, load_params

REPO = Path(__file__).resolve().parents[1]
PARAMS = load_params(REPO / PARAMS_PATH)
# The days of params.stats.days, and the day before the first of them, which is history only.
FIRST_STATS_DATE = datetime.datetime.strptime(str(PARAMS["stats"]["days"][0]), "%Y%m%d")
DAYS = [int((FIRST_STATS_DATE - datetime.timedelta(days=1)).strftime("%Y%m%d")), *PARAMS["stats"]["days"]]
# Rows per synthetic day: enough for every publisher to show up, small enough for the tests to take seconds.
ROWS_PER_DAY = 4000
