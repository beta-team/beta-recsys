|**[Installation](#installation)** |
**[Quick Start](#installation)** |
**[Documentation](https://beta-recsys.readthedocs.io/)** |
**[Contributing](#contributing)** |
**[Getting help](https://github.com/beta-team/community/blob/master/beta_recsys/README.md)** |
**[Citation](#Citation)**|

<p align="center">
  <a href="https://beta-recsys.readthedocs.io/">
    <img src="https://beta-recsys.readthedocs.io/en/latest/_static/Logo.svg" alt="Accord Project Logo" width="400">
  </a>
</p>

[![PyPI version](https://badge.fury.io/py/beta-rec.svg)](https://badge.fury.io/py/beta-rec)
[![Code Coverage](https://codecov.io/gh/leungyukshing/beta-recsys/branch/develop/graph/badge.svg)](https://codecov.io/gh/leungyukshing/beta-recsys)
[![CI](https://github.com/beta-team/beta-recsys/workflows/CI/badge.svg?branch=master)](https://github.com/beta-team/beta-recsys/actions?query=workflow%3ACI)
[![Documentation Status](https://readthedocs.org/projects/beta-recsys/badge/?version=stable)](http://beta-recsys.readthedocs.io/en/stable/)
[![GitHub](https://img.shields.io/badge/issue_tracking-github-blue.svg)](https://github.com/beta-team/beta-recsys/issues)
[![Slack Status](https://img.shields.io/badge/Join-Slack-purple)](https://join.slack.com/t/beta-recsys/shared_invite/zt-iwmlfb0g-yxeyzb0U9pZfFN~A4mrKpA)

Beta-RecSys an open source project for Building, Evaluating and Tuning Automated Recommender Systems.
Beta-RecSys aims to provide a practical data toolkit for building end-to-end recommendation systems in a standardized way.
It provided means for dataset preparation and splitting using common strategies, a generalized model engine for implementing recommender models using Pytorch with a lot of models available out-of-the-box,
as well as a unified training, validation, tuning and testing pipeline. Furthermore, Beta-RecSys is designed to be both modular and extensible, enabling new models to be quickly added to the framework.
It is deployable in a wide range of environments via pre-built docker containers and supports distributed parameter tuning using [Ray](https://github.com/ray-project/ray).

## Installation

### conda
If you use conda, you can install it with:
```shell
conda install beta-rec
```
### pip
If you use pip, you can install it with:
```shell
pip install beta-rec
```

### Docker

We also provide docker image for you to run this project on any platform. You can use the image with:

1. Pull image from Docker Hub

   ```
   docker pull betarecsys/beta-recsys:latest
   ```

2. Start a docker container with this image (Make sure the port 8888 is available on you local machine, or you can change the port in the command)

   ```
   docker run -ti --name beta-recsys -p 8888:8888 -d beta-recsys
   ```

3. Open Jupyter on a brower with this URL:

   ```
   http://localhost:8888
   ```
   
4. Enter `root` as the password for the notebook.

## Quick Start

### Downloading and Splitting Datasets

```python
from beta_rec.datasets.movielens import Movielens_100k
from beta_rec.data import BaseData
dataset = Movielens_100k()
split_dataset = dataset.load_leave_one_out(n_test=1)
data =  BaseData(split_dataset)
```

### Training model with MatrixFactorization

```python
config = {
    "config_file":"./configs/mf_default.json"
}
from beta_rec.recommenders import MatrixFactorization
model = MatrixFactorization(config)
model.train(data)
result = model.test(data.test[0])
```
where a default config josn file [./configs/mf_default.json](./configs/mf_default.json) will be loaded for traning the model.

### Tuning Model Hyper-parameters 

```python
config = {
    "config_file":"../configs/mf_default.json",
    "tune":True,
}
tune_result = model.train(data)
```

### Experiment with multiple models

```python
from beta_rec.recommenders import MatrixFactorization
from beta_rec.experiment.experiment import Experiment

# Initialise recommenders with their default configuration file

config = {
    "config_file":"configs/mf_default.json"
}

mf_1 = MatrixFactorization(config)
mf_2 = MatrixFactorization(config)

# Run experiments of the recommenders on the selected dataset

Experiment(
  datasets=[data],
  models=[mf_1, mf_2],
).run()
```
where the model will tune the hyper-parameters according to the specifed tuning scheme (e.g. [the default for MF](https://github.com/mengzaiqiao/beta-recsys/blob/master/configs/mf_default.json#L46)).

## Models

The following is a list of recommender models currently available in the repository, or to be implemented soon.

### General Models
  - [x] MF[![Example In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tJX4ZTtNp6tdGer-jUQ_ZZSIf9J2MB7G?usp=sharing): [Neural Collaborative Filtering vs. Matrix Factorization Revisited](https://arxiv.org/abs/2005.09683), arXiv’ 2020 
  - [x] GMF: Generalized Matrix Factorization, in [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031), WWW 2017
  - [x] MLP: Multi-Layer Perceptron, in [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031), WWW 2017
  - [x] NCF[![Example In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-3zfUNEexpB5eoTIwDfIqgMNFLQet2vV?usp=sharing): [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031),  WWW 2017
  - [x] CMN: [Collaborative memory network for recommendation systems](https://dl.acm.org/doi/abs/10.1145/3209978.3209991),  SIGIR 2018
  - [x] NGCF: [Neural graph collaborative filtering](https://dl.acm.org/doi/abs/10.1145/3331184.3331267), SIGIR 2019
  - [x] LightGCN: [**LightGCN**: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126), SIGIR 2020
  - [x] LCF: [Graph Convolutional Network for Recommendation with Low-pass Collaborative Filters](https://arxiv.org/abs/2006.15516)
  - [ ] VAECF: [Variational autoencoders for collaborative filtering](https://dl.acm.org/doi/abs/10.1145/3178876.3186150), WWW 2018

### Sequential Models
  - [x] NARM: [Neural Attentive Session-based Recommendation](https://arxiv.org/abs/1711.04725), CIKM 2017
  - [ ] Caser: [Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding
](https://dl.acm.org/doi/abs/10.1145/3159652.3159656), WSDM 2018
  - [ ] GRU4Rec: [Session-based recommendations with recurrent neural networks](https://arxiv.org/abs/1511.06939), ICLR 2016
  - [ ] SasRec:[**Self**-**attentive sequential recommendation**](https://ieeexplore.ieee.org/abstract/document/8594844/?casa_token=RINDZUuHnwoAAAAA:XBjSlh6-KqBjgCY1AWwgXyZqHtT_8zAPBMKjLIUJMlf6Ex9j55gG2UAsrRtG10roMUd6-_w3Jw). ICDM 2018
  - [ ] MARank: [Multi-Order Attentive Ranking Model for Sequential Recommendation
](https://ojs.aaai.org//index.php/AAAI/article/view/4516), AAAI 2019
  - [ ] NextItnet: [A Simple Convolutional Generative Network for Next Item Recommendation
](https://dl.acm.org/doi/abs/10.1145/3289600.3290975), WSDM 2019
  - [ ] BERT4Rec: [BERT4Rec: **Sequential recommendation** with **bidirectional encoder representations** from **transformer**](https://dl.acm.org/doi/abs/10.1145/3357384.3357895), CIKM 2019
  - [ ] TiSASRec: [Time Interval Aware Self-Attention for **Sequential Recommendation**](https://dl.acm.org/doi/abs/10.1145/3336191.3371786). WWW 2020

### Recommendation Models with Auxiliary information
  ### Baskets/Sessions
  - [x] Triple2vec: [Representing and recommending shopping baskets with complementarity, compatibility and loyalty](https://dl.acm.org/doi/abs/10.1145/3269206.3271786), CIKM 2018
  - [x] VBCAR: [Variational Bayesian Context-aware Representation for Grocery Recommendation](https://arxiv.org/abs/1909.07705),  arXiv’ 2019
  ### Knowledge Graph
  - [ ] KGAT: [Kgat: Knowledge graph attention network for recommendation](https://dl.acm.org/doi/abs/10.1145/3292500.3330989). SIGKDD 2019

> If you want your model to be implemented by our maintenance team (or by yourself), please submit an issue following our community [instruction]((#contributing)). 




## Recent Changing Logs ---> See [version release](https://github.com/beta-team/beta-recsys/releases).


## Contributing

This project welcomes contributions and suggestions. Please make sure to read the [Contributing Guide](https://github.com/beta-team/community/blob/master/beta_recsys/README.md) before creating a pull request. 
### Community meeting

- Meeting time: Saturday (1:30 – 2:30pm [UTC +0](https://24timezones.com/time-zone/utc#gref)) / (9:30 – 10:30pm [UTC +8](https://24timezones.com/time-zone/utc+8#gref)) [![Add Event](https://img.shields.io/badge/Add-Event-blue)](https://github.com/beta-team/community/releases/download/meeting/bi-weekly.meeting.ics)
- Meeting minutes: [notes](https://github.com/beta-team/community/tree/master/beta_recsys/meeting%20minutes)
- Meeting recordings: [recording links]: Can be found in each [meeting note](https://github.com/beta-team/community/tree/master/beta_recsys/meeting%20minutes).

### Discussion channels

- Slack: [![Slack Status](https://img.shields.io/badge/Join-Slack-purple)](https://join.slack.com/t/beta-recsys/shared_invite/zt-iwmlfb0g-yxeyzb0U9pZfFN~A4mrKpA)
- Mailing list: TBC

## Citation

If you use Beta-RecSys in you research, we would appreciate citations to the following paper:
```
@inproceedings{meng2020beta,
  title={BETA-Rec: Build, Evaluate and Tune Automated Recommender Systems},
  author={Meng, Zaiqiao and McCreadie, Richard and Macdonald, Craig and Ounis, Iadh and Liu, Siwei and Wu, Yaxiong and Wang, Xi and Liang, Shangsong and Liang, Yucheng and Zeng, Guangtao and others},
  booktitle={Fourteenth ACM Conference on Recommender Systems},
  pages={588--590},
  year={2020}
}
```

