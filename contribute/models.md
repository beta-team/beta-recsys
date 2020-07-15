# Add your own model

Our **Beta-RecSys** is an open source project for Building, Evaluating and Tuning Automated Recommender Systems. We are delight to see more recommender models to be added into our project. Therefore, our **beta-team** sincerely invite researchers and developers from the recommerdation community to contribute to our project. Below is a tutorial about how to intergrate an implementation of a new model, either a model you think is important or a novel model proposed by yourself.


## Code format
Our team is keen on keeping a nice, clean and documented code so that every single file can be understood by other researchers and developers. Therefore, shall we suggest those ,who would like to contribute to this project to follow the designed format. Indeed, those **Pull requests**, which do not follow the format can not pass the our **CI test**. We use *black-formatter, flake8-docstrings* and *isort* to format our code.

## Directory & file structure
```
Beta_Recsys 
│
└───beta_rec
│   └───core
│   │───data
│   │───datasets
│   │───utils
│   │───models
│    │└─── **new_model.py**
└───configs
│   └───**new_confg.json**
└───examples
│   └───**new_example.py**
```

In the file structure you may find the directory (in bold) where you need to add a new file for your model.
## Create a model in 3 steps

### 1.  create new_model.py
      
    from beta_rec.models.torch_engine import ModelEngine  
    from beta_rec.utils.common_util import print_dict_as_table, timeit  
      
      
    class NEWMODEL(torch.nn.Module):  
        """ A pytorch Module for a new model
     """  
      def __init__(self, config):  
            super(MF, self).__init__()  
            Parameters
			----------

		    References
		    ----------
      
        def forward(self, batch_data):  
            """  
      
			    Args: batch_data: tuple consists of (users, pos_items neg_items), which must be LongTensor  
			     Returns: 
		     """ 

            return scores
      
        def predict(self, users, items):  
            """ Model prediction
			     Args: users (int, or list of int):  user id(s) items (int, or list of int):  item id(s) Return: scores (int, or list of int): predicted scores of these user-item pairs """  users_t = torch.LongTensor(users).to(self.device)  
            items_t = torch.LongTensor(items).to(self.device)  
            with torch.no_grad():  
                scores = 
            return scores  
      
      
    class NEWMODELEngine(ModelEngine):  
        def __init__(self, config):  

      
        def train_single_batch(self, batch_data):  
            """ Train a single batch  
      
   
        @timeit  
      def train_an_epoch(self, train_loader, epoch_id):  
            
		      for batch_data in train_loader:  
                loss, reg = self.train_single_batch(batch_data)  
                total_loss += loss  
                regularizer += reg  
            print(f"[Training Epoch {epoch_id}], Loss {loss}, Regularizer {regularizer}")  
            self.writer.add_scalar("model/loss", total_loss, epoch_id)  
            self.writer.add_scalar("model/regularizer", regularizer, epoch_id)
In the new_model.py, you may want to add two classes, class **NEWMODEL** (all in capital) and class **NEWMODELEngine**. The NEWMODEL calss should include all necessary initialisations (e.g. embeddings initialisation), *forward function* to calculate all intermedinate variables and *predict function* to calculate predicted scores for each (user, item) pair. In the NEWMODELEngine, first you need load the training data and corresponding configs. Then you use two functions *train_an_epoch* and *train_single_batch* to feed data to the **NEWMODEL** class. A classic train_loader, which can sample user, positive items and negative items is already included in our project. You can see much efforts by loading existing functions.
### 2.  create new_default.json

You also need a .json file, which includes all parameters for your models. This config file bring much convenience when you want to run a model several times with different parameters. Parameters can be changed from the command line. Below is a exmaple of a config file for the matrix factorisation model.

    {  
        "system": {  
            "root_dir": "../",  
            "log_dir": "logs/",  
            "result_dir": "results/",  
            "process_dir": "processes/",  
            "checkpoint_dir": "checkpoints/",  
            "dataset_dir": "datasets/",  
            "run_dir": "runs/",  
            "tune_dir": "tune_results/",  
            "device": "gpu",  
            "seed": 2020,  
            "metrics": ["ndcg", "precision", "recall", "map"],  
            "k": [5,10,20],  
            "valid_metric": "ndcg",  
            "valid_k": 10,  
            "result_file": "mf_result.csv"  
      },  
        "dataset": {  
            "dataset": "ml_100k",  
            "data_split": "leave_one_out",  
            "download": false,  
            "random": false,  
            "test_rate": 0.2,  
            "by_user": false,  
            "n_test": 10,  
            "n_negative": 100,  
            "result_col": ["dataset","data_split","test_rate","n_negative"]  
        },  
        "model": {  
            "model": "MF",  
            "config_id": "default",  
            "emb_dim": 64,  
            "num_negative": 4,  
            "batch_size": 400,  
            "batch_eval": true,  
            "dropout": 0.0,  
            "optimizer": "adam",  
            "loss": "bpr",  
            "lr": 0.05,  
            "reg": 0.001,  
            "max_epoch": 20,  
            "save_name": "mf.model",  
            "result_col": ["model","emb_dim","batch_size","dropout","optimizer","loss","lr","reg"]  
        },  
        "tunable": [  
            {"name": "loss", "type": "choice", "values": ["bce", "bpr"]}  
        ]  
    }

### 3.  create new_example.py

    """  
     isort:skip_file"""  
    import argparse  
    import os  
    import sys  
    import time  
      
    sys.path.append("../")  
      
    from ray import tune  
      
    from beta_rec.core.train_engine import TrainEngine  
    from beta_rec.models.new_model import NEWMODELEngine  
    from beta_rec.utils.common_util import DictToObject, str2bool  
    from beta_rec.utils.monitor import Monitor  
      
      
    def parse_args():  
        """ Parse args from command line  
      
     Returns: args object. """  parser = argparse.ArgumentParser(description="Run new model..")  
        parser.add_argument(  
            "--config_file",  
            nargs="?",  
            type=str,  
            default="../configs/new_default.json",  
            help="Specify the config file name. Only accept a file from ../configs/",  
        )  
        parser.add_argument(  
            "--root_dir", nargs="?", type=str, help="Root path of the project",  
        )  
        # If the following settings are specified with command line,  
     # These settings will used to update the parameters received from the config file.  parser.add_argument(  
            "--dataset",  
            nargs="?",  
            type=str,  
            help="Options are: tafeng, dunnhunmby and instacart",  
        )  
        parser.add_argument(  
            "--data_split",  
            nargs="?",  
            type=str,  
            help="Options are: leave_one_out and temporal",  
        )  
        parser.add_argument(  
            "--tune", nargs="?", type=str2bool, help="Tun parameter",  
        )  
        parser.add_argument(  
            "--device", nargs="?", type=str, help="Device",  
        )  
        parser.add_argument(  
            "--loss", nargs="?", type=str, help="loss: bpr or bce",  
        )  
        parser.add_argument(  
            "--remark", nargs="?", type=str, help="remark",  
        )  
        parser.add_argument(  
            "--emb_dim", nargs="?", type=int, help="Dimension of the embedding."  
      )  
        parser.add_argument("--lr", nargs="?", type=float, help="Initial learning rate.")  
        parser.add_argument("--reg", nargs="?", type=float, help="regularization.")  
        parser.add_argument("--max_epoch", nargs="?", type=int, help="Number of max epoch.")  
        parser.add_argument(  
            "--batch_size", nargs="?", type=int, help="Batch size for training."  
      )  
        return parser.parse_args()  
      
      
    class NEW_train(TrainEngine):  
        def __init__(self, args):  
            print(args)  
            super(NEW_train, self).__init__(args)  
      
        def train(self):  
            self.load_dataset()  
            self.gpu_id, self.config["device_str"] = self.get_device()  
            """ Main training navigator  
      
     Returns:  
      self.monitor = Monitor(  
                log_dir=self.config["system"]["run_dir"], delay=1, gpu_id=self.gpu_id  
            )  
            if self.config["model"]["loss"] == "bpr":  
                train_loader = self.data.instance_bpr_loader(  
                    batch_size=self.config["model"]["batch_size"],  
                    device=self.config["model"]["device_str"],  
                )  
            elif self.config["model"]["loss"] == "bce":  
                train_loader = self.data.instance_bce_loader(  
                    num_negative=self.config["model"]["num_negative"],  
                    batch_size=self.config["model"]["batch_size"],  
                    device=self.config["model"]["device_str"],  
                )  
            else:  
                raise ValueError(  
                    f"Unsupported loss type {self.config['loss']}, try other options: 'bpr' or 'bce'"  
      )  
      
            self.engine = NEWMODELEngine(self.config)  
            self.model_save_dir = os.path.join(  
                self.config["system"]["model_save_dir"], self.config["model"]["save_name"]  
            )  
            self._train(self.engine, train_loader, self.model_save_dir)  
            self.config["run_time"] = self.monitor.stop()  
            return self.eval_engine.best_valid_performance  
      
      
    def tune_train(config):  
        """Train the model with a hypyer-parameter tuner (ray)  
      
     Args: config (dict): All the parameters for the model  
     Returns:  
     """  
	    train_engine = NEW_train(DictToObject(config))  
        best_performance = train_engine.train()  
        train_engine.test()  
        while train_engine.eval_engine.n_worker > 0:  
            time.sleep(20)  
        tune.track.log(valid_metric=best_performance)  
      
      
    if __name__ == "__main__":  
        args = parse_args()  
        if args.tune:  
            train_engine = NEW_train(args)  
            train_engine.tune(tune_train)  
        else:  
            train_engine = NEW_train(args)  
            train_engine.train()  
            train_engine.test()


In this new_example.py file, you need import the TrainEngine from core and the NEWMODELEngine from the new_model.py. The parse_args function will help you to load parameters from the command line and the config file. You can simply run your model once or you may want to apply a grid search by the Tune module. You should define all tunable parameters in your config file.
