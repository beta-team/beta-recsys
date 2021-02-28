import pathlib
from glob import glob

import pkg_resources
from setuptools import find_packages, setup

__version__ = "0.3.2" ""
url = "https://github.com/beta-team/beta-recsys"

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text(encoding="utf-8")

with pathlib.Path("requirements.txt").open(encoding="utf-8") as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_txt)
    ]

setup_requires = ["pytest-runner", "isort"]
tests_require = ["pytest", "pytest-cov", "mock"]

setup(
    name="beta_rec",
    version=__version__,
    description="Beta-RecSys: Build, Evaluate and Tune Automated Recommender Systems",
    long_description=README,
    long_description_content_type="text/markdown",
    author_email="recsys.beta@gmail.com",
    url=url,
    download_url="{}/archive/{}.tar.gz".format(url, __version__),
    keywords=["pytorch", "recommender system", "recommendations"],
    python_requires=">=3.6.9",
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=find_packages(),
    data_files=[("beta_rec", glob("configs/*"))],  # source_dir only - not recursive
)
