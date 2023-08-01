from typing import Literal
from enum import Enum

from KImie.utils.sys import set_user_folder, get_user_folder
from KImie.utils.logging import KIMIE_LOGGER
from KImie.exceptions import *


class KImieMode(Enum):
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    TEST = "test"


MODE: KImieMode = KImieMode.PRODUCTION
