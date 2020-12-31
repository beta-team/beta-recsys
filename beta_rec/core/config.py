import os
import platform
import sys

# OS constants (some libraries/features are OS-dependent)
BSD = sys.platform.find("bsd") != -1
LINUX = sys.platform.startswith("linux")
MACOS = sys.platform.startswith("darwin")
SUNOS = sys.platform.startswith("sunos")
WINDOWS = sys.platform.startswith("win")
WSL = (
    "linux" in platform.system().lower() and "microsoft" in platform.uname()[3].lower()
)


def user_config_dir():
    r"""Return the per-user config dir (full path).
    - Linux, *BSD, SunOS: ~/.config/beta_rec
    - macOS: ~/Library/Application Support/beta_rec
    - Windows: %APPDATA%\beta_rec
    """
    if WINDOWS:
        path = os.environ.get("APPDATA")
    elif MACOS:
        path = os.path.expanduser("~/Library/Application Support")
    else:
        path = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    if path is None:
        path = ""
    else:
        path = os.path.join(path, "beta_rec")

    return path


def system_config_dir():
    r"""Return the system-wide config dir (full path).
    - Linux, SunOS: /etc/beta_rec
    - *BSD, macOS: /usr/local/etc/beta_rec
    - Windows: %APPDATA%\beta_rec
    """
    if LINUX or SUNOS:
        path = "/etc"
    elif BSD or MACOS:
        path = "/usr/local/etc"
    else:
        path = os.environ.get("APPDATA")
    if path is None:
        path = ""
    else:
        path = os.path.join(path, "beta_rec")

    return path


def find_config(config_file):
    """Read the config file, if it exists. Using defaults otherwise."""
    if os.path.exists(config_file):
        return config_file
    for config_file in config_file_paths(config_file):
        print("Search default config file in {}".format(config_file))
        if os.path.exists(config_file):
            print("Found default config file in {}".format(config_file))
            return config_file

def config_file_paths(config_file):
    r"""Get a list of config file paths.
    The list is built taking into account of the OS, priority and location.
    * custom path: /path/to/beta_rec
    * Linux, SunOS: ~/.config/beta_rec, /etc/beta_rec
    * *BSD: ~/.config/beta_rec, /usr/local/etc/beta_rec
    * macOS: ~/Library/Application Support/beta_rec, /usr/local/etc/beta_rec
    * Windows: %APPDATA%\glances
    The config file will be searched in the following order of priority:
        * /path/to/file (via -C flag)
        * user's home directory (per-user settings)
        * system-wide directory (system-wide settings)
    """
    config_filename = config_file.replace("../", "beta_rec/")
    paths = []
    paths.append(os.path.join(user_config_dir(), config_filename))
    paths.append(os.path.join(system_config_dir(), config_filename))

    return paths
