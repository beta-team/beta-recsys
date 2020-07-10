from setuptools import find_packages, setup

__version__ = "0.1.0"
url = "https://github.com/beta-team/beta-recsys"

install_requires = [
    "requests~=2.23.0",
    "prometheus_client~=0.8.0",
    "tensorboardX~=2.0",
    "ray~=0.8.5",
    "torch==1.4.0",
    "numpy==1.18.1",
    "pytest==5.4.1",
    "flake8==3.7.9",
    "pytest-cov==2.8.1",
    "pandas==1.0.3",
    "mock==4.0.1",
    "scipy==1.4.1",
    "scikit-learn==0.22.2.post1",
    "gputil==1.4.0",
    "aiofiles~=0.4.0",
    "aiohttp~=3.6.2",
    "nest_asyncio~=1.3.3",
    "livelossplot~=0.5.0",
    "cornac~=1.6.1",
    "py-cpuinfo~=5.0.0",
    "psutil~=5.7.0",
    "tabulate~=0.8.7",
    "py7zr~=0.6",
    "flake8-black==0.1.2",
    "wheel~=0.34.2",
    "sphinx_markdown_tables~=0.0.14",
    "flake8-isort~=3.0.1",
    "isort==4.3.20",
    "ax-platform==0.1.13",
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
