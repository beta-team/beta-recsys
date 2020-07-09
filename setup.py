from setuptools import find_packages, setup

__version__ = "0.1.0"
url = "https://github.com/beta-team/beta-recsys"

install_requires = [
    "requests",
    "prometheus_client",
    "tensorboardX",
    "ray",
    "torch",
    "numpy",
    "tqdm",
    "pytest",
    "flake8",
    "pytest-cov",
    "pandas",
    "mock",
    "scipy",
    "scikit-learn",
    "gputil",
    "aiofiles",
    "aiohttp",
    "nest_asyncio",
    "livelossplot",
    "cornac",
    "py-cpuinfo",
    "psutil",
    "tabulate",
    "py7zr",
    "flake8-black",
    "wheel",
    "sphinx_markdown_tables",
    "isort",
]
setup_requires = ["pytest-runner", "isort"]
tests_require = ["pytest", "pytest-cov", "mock"]

setup(
    name="beta-recsys",
    version=__version__,
    description="Beta-RecSys: Beta-RecSys: Build, Evaluate and Tune Automated Recommender Systems",
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
