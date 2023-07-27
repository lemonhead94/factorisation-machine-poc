from typing import Tuple

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset


# Bipartie Graph Data Loader for Dish and Ingredient Interaction Data
# will be used as a Graph Embedding for the Factorization Machine
class DishIngredientDataset(
    Dataset[Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]]
):
    def __init__(self, data: npt.NDArray[np.int32], dims: npt.NDArray[np.int32]):
        """
        Dataloader for dish and ingredients.
        """
        super(DishIngredientDataset, self).__init__()
        self.interactions = data
        self.dims = dims

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(
        self, index: int
    ) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """
        Return the pairs of dish & ingredient and the dimensions (n_dish, n_ingredient).
        """
        return self.interactions[index][:-1], self.dims
