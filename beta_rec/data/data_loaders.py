from torch.utils.data import Dataset


class RatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset."""

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """Init UserItemRatingDataset Class.

        Args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair.
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        """Get an item from dataset."""
        return (
            self.user_tensor[index],
            self.item_tensor[index],
            self.target_tensor[index],
        )

    def __len__(self):
        """Get the size of the dataset."""
        return self.user_tensor.size(0)


class PairwiseNegativeDataset(Dataset):
    """Wrapper, convert <user, pos_item, neg_item> Tensor into Pytorch Dataset."""

    def __init__(self, user_tensor, pos_item_tensor, neg_item_tensor):
        """Init PairwiseNegativeDataset Class.

        Args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair.
        """
        self.user_tensor = user_tensor
        self.pos_item_tensor = pos_item_tensor
        self.neg_item_tensor = neg_item_tensor

    def __getitem__(self, index):
        """Get an item from the dataset."""
        return (
            self.user_tensor[index],
            self.pos_item_tensor[index],
            self.neg_item_tensor[index],
        )

    def __len__(self):
        """Get the size of the dataset."""
        return self.user_tensor.size(0)
