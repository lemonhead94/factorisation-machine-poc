import pandas as pd
from torch.utils.data import Dataset


class DishIngredientDataset(Dataset):
    def __init__(self, data: pd.DataFrame, dims):
        """
        Dataloader for dish and ingredients.
        """
        super(DishIngredientDataset, self).__init__()
        self.interactions = data
        self.dims = dims

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, index):
        """
        Return the pairs user-item and the target.
        """
        return self.interactions[index][:-1], self.interactions[index][-1]
