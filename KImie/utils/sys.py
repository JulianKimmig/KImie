import logging
import os

# sets the user dir to a hidden dir in the home dir
if "KIMIE_DIR" not in os.environ:
    os.environ["KIMIE_DIR"] = os.path.join(os.path.expanduser("~"), ".KImie")

# assert that the KIMIE_DIR exists
os.makedirs(os.environ["KIMIE_DIR"], exist_ok=True)

# the .env file stays in the ~/.KImie flolder regardless of the KIMIE_DIR env var
_ENV_FILE = os.path.join(os.environ["KIMIE_DIR"], ".env")

# dictionary contains the environment variables usually all items in this dict are also in os.environ
LOKAL_ENVS = {}

# store changelistener that are called when the user dir changes
_USERFOLDERCHANGELISTENER = []


def _read_env():
    """
    reads the environment variables from the .env file and updates the os.environ dictionary
    """

    # if the .env file does not exist, create it
    if not os.path.exists(_ENV_FILE):
        with open(_ENV_FILE, "w+"):
            pass
    # read the .env file
    with open(_ENV_FILE, "r") as f:
        cont = f.read()

    # reads the environment variables to the LOKAL_ENVS dictionary
    for l in cont.split("\n"):
        try:
            k, v = l.split("=", 1)
            LOKAL_ENVS[k] = v
        except Exception:
            pass
    # update the os.environ dictionary
    for k, v in LOKAL_ENVS.items():
        os.environ[k] = v


def _write_env() -> None:
    """
    writes the environment variables to the .env file
    """
    cont = "\n".join([f"{k}={v}" for k, v in LOKAL_ENVS.items()]) + "\n"
    with open(_ENV_FILE, "w+") as f:
        f.write(cont)


# read the env file to the os.eviron dict
_read_env()


def set_environment_variable(key: str, value: str, permanent=False) -> None:
    """
    sets an environment variable. If the variable is permanent, it will be written to the .env file
    """
    key, value = str(key), str(value)
    os.environ[key] = value
    if permanent:
        LOKAL_ENVS[key] = os.environ[key]
        _write_env()


def set_user_folder(path: str, permanent: bool = False) -> None:
    """
    sets the user folder. Note that the original folder will remain with at least an information file containing all the configuration
    :param path: new user folder
    :param permanent: if true, the new folder will be set permanently as the user folder (default: False)
    """
    set_environment_variable("KIMIE_DIR", path, permanent=permanent)
    os.makedirs(os.environ["KIMIE_DIR"], exist_ok=True)

    # call the changelistener
    for cl in _USERFOLDERCHANGELISTENER:
        cl(get_user_folder())


def get_user_folder() -> str:
    """
    returns the user folder
    """
    return os.environ["KIMIE_DIR"]


def get_temp_dir() -> str:
    """
    returns the KImie temp folder
    """
    import tempfile

    return os.path.join(tempfile.gettempdir(), "KImie")


def enter_test_mode():
    # set the user folder to the temp folder
    set_user_folder(get_temp_dir())
    # logger in debug mode
    from KImie import KIMIE_LOGGER

    KIMIE_LOGGER.setLevel(logging.DEBUG)
