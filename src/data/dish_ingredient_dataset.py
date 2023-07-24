import numpy as np
from torch.utils.data import Dataset


# Bipartie Graph Data Loader for Dish and Ingredient Interaction Data
# will be used as a Graph Embedding for the Factorization Machine
class DishIngredientDataset(Dataset):
    def __init__(self, data: np.ndarray, dims: np.ndarray):
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
        Return the pairs dish, ingredient, and the dimensions.
        """
        return self.interactions[index][:-1], self.interactions[index][-1], self.dims
