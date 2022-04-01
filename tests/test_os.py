import datetime
import os
import time
import unittest

from KImie.utils.logging import get_logs_as_dataframe, KIMIE_LOGGER
from KImie.utils.sys import get_temp_dir, set_user_folder, get_user_folder


class OSTest(unittest.TestCase):
    def setUp(self) -> None:
        set_user_folder(get_temp_dir(), permanent=False)
        KIMIE_LOGGER.setLevel("DEBUG")

    def test_logging(self):
        t = datetime.datetime.now()
        KIMIE_LOGGER.debug(f"logging test")
        df = get_logs_as_dataframe()
        assert abs(t - df.iloc[-1]["time"]) < datetime.timedelta(
            milliseconds=100
        ), "Logging time is not correct"
        time.sleep(0.1)
        KIMIE_LOGGER.setLevel("INFO")
        t = datetime.datetime.now()
        KIMIE_LOGGER.debug(f"logging test")
        df = get_logs_as_dataframe()
        assert abs(t - df.iloc[-1]["time"]) > datetime.timedelta(
            milliseconds=100
        ), "logging should not be logged"

    def test_change_home_dir(self):
        set_user_folder(get_temp_dir() + "_bu", permanent=False)
        self.assertEqual(get_user_folder(), get_temp_dir() + "_bu")
        self.assertEqual(os.environ["KIMIE_DIR"], get_temp_dir() + "_bu")
