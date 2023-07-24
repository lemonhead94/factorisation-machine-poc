import numpy as np
from torch.utils.data import Dataset


# Bipartie Graph Data Loader for Ingredient and User Interaction Data
# will be used as a Graph Embedding for the Factorization Machine
class UserIngredientDataset(Dataset):
    def __init__(self, data: np.ndarray, dims: np.ndarray):
        """
        Dataloader for user and ingredient interactions.
        """
        super(UserIngredientDataset, self).__init__()
        self.interactions = data
        self.dims = dims

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, index):
        """
        Return the pairs user, ingredient, number of views, and the dimensions.
        """
        user_ingredient_views = self.interactions[index][:-1]
        num_views = self.interactions[index][-1]
        return user_ingredient_views, num_views, self.dims
