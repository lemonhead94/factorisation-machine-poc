import pandas as pd
from torch.utils.data import Dataset


# Bipartie Graph Data Loader for Ingredient and User Interaction Data
# will be use as a Graph Embedding for the Factorisation Machine
class IngredientUserDataset(Dataset):
    def __init__(self, data: pd.DataFrame, dims):
        """
        Dataloader for dish and ingredients.
        """
        super(IngredientUserDataset, self).__init__()
        self.interactions = data
        self.dims = dims

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, index):
        """
        Return the pairs user-item and the target.
        """
        return self.interactions[index][:-1], self.interactions[index][-1]
