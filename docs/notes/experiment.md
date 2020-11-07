# Experiment

Beta-Rec platform provides a convenient experiment interface to use for running a series of experiments to examine models' performances on datasets with various data split setups.
![](https://github.com/beta-team/beta-recsys/blob/wangxieric-patch-1/docs/_static/img/experiment_pipe.pdf)

We use the Matrix Factorisation-based recommender as example and train and test it on the MovieLens-100k dataset, which adopt the leave one out data split setup. The example notebook is also provided [here](https://github.com/beta-team/beta-recsys/blob/master/examples/experiment.ipynb).

In the following content, we illustrate our experimental pipeline with the aforementioned example step by step:

## Load dataset

First, we initiase the BaseData with an available dataset and its data split setup from the platform. You can found the availble datasets and data split functions [here](https://beta-recsys.readthedocs.io/en/latest/notes/datasets.html). An example is also given as follows: 

```python
import sys
sys.path.append("../")
from beta_rec.datasets.movielens import Movielens_100k
from beta_rec.data.base_data import BaseData

# Initialise dataset and the corresponding data split strategy

dataset = Movielens_100k()
split_dataset = dataset.load_leave_one_out(n_test=1)
data = BaseData(split_dataset)
```

## Model Configuration

Next, we select an targeted models to conduct the experiments. In particular, each model has its default and corresponding configuration file, which is listed in the [configs](https://github.com/beta-team/beta-recsys/tree/master/configs) folder. There are two options to update the configuration of the selected models:

(1) Update the configuration file (e.g. mf_default.json).

(2) Update the default values of instance variables of the Experiment class (e.g. eval_scopes).

The model configuration of two MF models can be written as follows:

```python
from beta_rec.recommenders import MatrixFactorization
from beta_rec.experiment.experiment import Experiment

# Initialise recommenders with their default configuration files

config1 = {
    "config_file":"configs/mf_default.json"
}

config2 = {
    "config_file":"configs/mf_default.json"
}

mf_1 = MatrixFactorization(config1)
mf_2 = MatrixFactorization(config2)
```

## Run Experiment

After initialising the selected dataset and the models, we can pass these two objects to the experiment class and run experiments as follows:

```python
# Run experiments of the recommenders on the selected dataset

Experiment(
  datasets=[data],
  models=[mf_1, mf_2],
).run()
```





