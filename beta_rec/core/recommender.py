from munch import munchify

from .train_engine import TrainEngine


class Recommender(TrainEngine):
    """The recommender base class."""

    def __init__(self, config, name=""):
        """Initialize model config.

        Args:
            config: model config.
            name: the name of the Recommender.
        """
        self.name = name
        config = munchify(config)
        super(Recommender, self).__init__(config)

    def train(self, train_df, valid_df=None, test_df=None):
        """Train a model, need to be implement for each model.

        Args:
            train_df: The input DataFrame training dataset.
            valid_df: The validation dataset.
            test_df: The testing datasets.

        Returns:
            dict: A dictionary of validation performance result.
        """
        pass

    def test(self, test_df):
        """Score and Evaluate for a dataframe data.

        Args:
            test_df: The input DataFrame user-item pairs.

        Returns:
            None

        """
        result = self.eval_engine.test_eval(test_df, self.engine.model)
        return result

    def load(self, model_dir):
        """Load a trained model.

        Args:
            model_dir: model saving path.

        Returns:
            Recommender: loaded model

        """
        self.engine.resume_checkpoint(model_dir)

    def predict(self, data_df):
        """Predict scores for a input DataFrame user-item pairs.

        Args:
            data_df: The input DataFrame user-item pairs.

        Returns:
            numpy array: the predict score for the user-item pairs

        """
        return self.eval_engine.predict(data_df, self.engine.model)
