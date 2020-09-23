import pathlib

import pkg_resources
from setuptools import find_packages, setup

__version__ = "0.2.1"
url = "https://github.com/beta-team/beta-recsys"


with pathlib.Path("requirements.txt").open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]

setup_requires = ["pytest-runner", "isort"]
tests_require = ["pytest", "pytest-cov", "mock"]

setup(
    name="beta-recsys",
    version=__version__,
    description=(
        "Beta-RecSys: Beta-RecSys: Build, Evaluate and Tune Automated Recommender"
        " Systems"
    ),
    author_email="recsys.beta@gmail.com",
    url=url,
    download_url="{}/archive/{}.tar.gz".format(url, __version__),
    keywords=["pytorch", "recommender system", "recommendations"],
    python_requires=">=3.7",
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=find_packages(),
)
