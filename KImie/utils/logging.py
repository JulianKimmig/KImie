import csv
import io
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

import pandas as pd

from KImie.utils.sys import (
    get_user_folder,
    _USERFOLDERCHANGELISTENER,
    set_environment_variable,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# sets the matplotlib logger to warning
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# set the rdkit logger to critical
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# set the numba logger to warning
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


# create KImie logger
KIMIE_LOGGER = logging.getLogger("KImie")

os.makedirs(os.path.join(get_user_folder(), "logs"), exist_ok=True)


class CsvFormatter(logging.Formatter):
    def __init__(self):
        super().__init__("%(asctime)s - %(levelname)s - %(message)s")
        self.output = io.StringIO()
        self.writer = csv.writer(self.output, quoting=csv.QUOTE_ALL)

    def format(self, record):
        super().format(record)
        self.writer.writerow([record.asctime, record.levelname, record.msg])
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()


def _add_filehandler():
    if hasattr(KIMIE_LOGGER, "_kimielogger_filehandler"):
        if KIMIE_LOGGER._kimielogger_filehandler is not None:
            KIMIE_LOGGER.removeHandler(KIMIE_LOGGER._kimielogger_filehandler)
            if isinstance(KIMIE_LOGGER._kimielogger_filehandler, RotatingFileHandler):
                KIMIE_LOGGER._kimielogger_filehandler.close()

    filehandler = RotatingFileHandler(
        os.path.join(get_user_folder(), "logs", "KImie.csv"),
        maxBytes=2**20,
        backupCount=10,
    )
    filehandler.setLevel(KIMIE_LOGGER.level)
    # create formatter and add it to the handlers
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    filehandler.setFormatter(CsvFormatter())
    # add the handlers to the logger
    KIMIE_LOGGER.addHandler(filehandler)
    KIMIE_LOGGER._kimielogger_filehandler = filehandler


_add_filehandler()

try:
    # link colored logs for colorful output
    import coloredlogs

    coloredlogs.install(level=KIMIE_LOGGER.level, logger=KIMIE_LOGGER)
except ModuleNotFoundError:
    pass

# extend the set_level function to also set the level of the filehandler
_ml_set_level = KIMIE_LOGGER.setLevel


def set_level(level: int = logging.INFO, permanent=False):
    _ml_set_level(level)
    level = KIMIE_LOGGER.level
    if hasattr(KIMIE_LOGGER, "_kimielogger_filehandler"):
        KIMIE_LOGGER._kimielogger_filehandler.setLevel(level)
    set_environment_variable("KIMIE_LOGGER_LEVEL", str(level), permanent=permanent)


KIMIE_LOGGER.setLevel = set_level

# set the logging level to the level in the environment variable
KIMIE_LOGGER.setLevel(int(os.environ.get("KIMIE_LOGGER_LEVEL", logging.INFO)))


# if the user folder has changed, update the filehandler
def _on_dir_change(d):
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    _add_filehandler()


_USERFOLDERCHANGELISTENER.append(_on_dir_change)


def get_logs_as_dataframe():
    d = []
    if not hasattr(KIMIE_LOGGER, "_kimielogger_filehandler"):
        KIMIE_LOGGER._kimielogger_filehandler = None
        return pd.DataFrame(columns=["time", "level", "message"])

    basefile = KIMIE_LOGGER._kimielogger_filehandler.baseFilename
    basename = os.path.basename(basefile)
    logdir = os.path.dirname(basefile)
    sdfs = []
    for fn in os.listdir(logdir):
        if not fn.startswith(basename):
            continue
        # if not fn.endswith(".log"):
        #     continue

        sdf = pd.read_csv(
            os.path.join(logdir, fn),
            index_col=None,
            header=None,
            names=["time", "level", "message"],
        )
        sdfs.append(sdf)

    df = pd.concat(sdfs)
    # df=pd.DataFrame(d,columns=["time","level","message"])
    df["message"] = df["message"].apply(lambda x: x.strip())
    df["time"] = pd.to_datetime(df["time"])
    df["idx"] = df.index
    df.sort_values(by=["time", "idx"], inplace=True, ignore_index=True)
    df.drop(["idx"], axis=1, inplace=True)

    return df
