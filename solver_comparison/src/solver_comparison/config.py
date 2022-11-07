import os


def _key(name: str) -> str:
    return f"KMEREXPR_BENCH_{name}"


K_WORKSPACE = _key("WORKSPACE")
K_LOGLEVEL = _key("LOGLEVEL")


def _get_env_var(key: str) -> str:
    val = os.environ.get(key, None)
    if val is None:
        raise EnvironmentError(
            f"Environment variable {key} undefined. "
            f"See readme for how to set environment variable."
        )
    return val


def workspace() -> str:
    return os.path.realpath(_get_env_var(K_WORKSPACE))


def log_dir() -> str:
    return os.path.join(workspace(), "logs")


def datalog_dir() -> str:
    return os.path.join(workspace(), "data_logs")


def get_console_logging_level():
    return _get_env_var(K_LOGLEVEL)


def experiment_dir():
    return os.path.join(workspace(), "results")
