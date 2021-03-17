import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class ModelEngine(object):
    """Meta Engine for training & evaluating NCF model.

    Note: Subclass should implement self.model!
    """

    def __init__(self, config):
        """Initialize ModelEngine Class."""
        self.config = config  # model configuration, should be a dic
        self.set_device()
        self.set_optimizer()
        self.model.to(self.device)
        print(self.model)
        self.writer = SummaryWriter(
            log_dir=config["system"]["run_dir"]
        )  # tensorboard writer

    def set_optimizer(self):
        """Set optimizer in the model."""
        if self.config["model"]["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config["model"]["lr"],
            )
        elif self.config["model"]["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["model"]["lr"],
            )
        elif self.config["model"]["optimizer"] == "rmsprop":
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.config["model"]["lr"],
            )

    def set_device(self):
        """Set device."""
        self.device = torch.device(self.config["model"]["device_str"])
        self.model.device = self.device
        print("Setting device for torch_engine", self.device)

    def train_single_batch(self, batch_data, ratings=None):
        """Train the model in a single batch."""
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.optimizer.zero_grad()
        ratings_pred = self.model.forward(batch_data)
        loss = self.model.loss(ratings_pred.view(-1), ratings)
        loss.backward()
        self.model.optimizer.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        """Train the model in one epoch."""
        assert hasattr(self, "model"), "Please specify the exact model !"
        self.model.train()
        total_loss = 0
        for batch_id, batch_data in enumerate(train_loader):
            assert isinstance(batch_data, torch.LongTensor)
            loss = self.train_single_batch(batch_data)
            total_loss += loss
        print("[Training Epoch {}], Loss {}".format(epoch_id, total_loss))
        self.writer.add_scalar("model/loss", total_loss, epoch_id)

    def save_checkpoint(self, model_dir):
        """Save checkpoint."""
        assert hasattr(self, "model"), "Please specify the exact model !"
        torch.save(self.model.state_dict(), model_dir)

    # to do
    def resume_checkpoint(self, model_dir, model=None):
        """Resume model with checkpoint."""
        assert hasattr(self, "model"), "Please specify the exact model !"
        print("loading model from:", model_dir)
        state_dict = torch.load(
            model_dir, map_location=self.device
        )  # ensure all storage are on gpu
        if model is None:
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            return self.model
        else:
            model.load_state_dict(state_dict)
            model.to(self.device)
            return model

    def bpr_loss(self, pos_scores, neg_scores):
        """Bayesian Personalised Ranking (BPR) pairwise loss function.

        Note that the sizes of pos_scores and neg_scores should be equal.

        Args:
            pos_scores (tensor): Tensor containing predictions for known positive items.
            neg_scores (tensor): Tensor containing predictions for sampled negative items.

        Returns:
            loss.
        """
        maxi = F.logsigmoid(pos_scores - neg_scores)
        loss = -torch.mean(maxi)
        return loss

    def bce_loss(self, scores, ratings):
        """Binary Cross-Entropy (BCE) pointwise loss, also known as log loss or logistic loss.

        Args:
            scores (tensor): Tensor containing predictions for both positive and negative items.
            ratings (tensor): Tensor containing ratings for both positive and negative items.

        Returns:
            loss.
        """
        # Calculate Binary Cross Entropy loss
        criterion = torch.nn.BCELoss()
        loss = criterion(scores, ratings)
        return loss
