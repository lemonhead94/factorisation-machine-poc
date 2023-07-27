from typing import Tuple

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset


# Bipartie Graph Data Loader for Ingredient and User Interaction Data
# will be used as a Graph Embedding for the Factorization Machine
class UserIngredientDataset(
    Dataset[Tuple[npt.NDArray[np.int32], np.int32, npt.NDArray[np.int32]]]
):
    def __init__(self, data: npt.NDArray[np.int32], dims: npt.NDArray[np.int32]):
        """
        Dataloader for user and ingredient interactions.
        """
        super(UserIngredientDataset, self).__init__()
        self.interactions = data
        self.dims = dims

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(
        self, index: int
    ) -> Tuple[npt.NDArray[np.int32], np.int32, npt.NDArray[np.int32]]:
        """
        Return user_ingredient_views pair, number of views, and the dimensions.
        """
        return self.interactions[index][:-1], self.interactions[index][-1], self.dims
