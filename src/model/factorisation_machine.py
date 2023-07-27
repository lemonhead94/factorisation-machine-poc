from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.model.gce_layer import GCELayerDishIngredient, GCELayerUserDish


# Graph-Aware Recommender Systems with Heterogeneous Graph Fusion for Enhanced Ingredient Recommendations
# - Graph-Aware Recommender System: a RS that incorporates graph-based techniques to model user-item interactions and item-item relationships. Interactions involve users, dishes, and ingredients, forming a heterogeneous information network.
# - Heterogeneous Graph Fusion: multiple graphs or different types of interactions in the heterogeneous information network. These interactions include user-dish interactions, dish-ingredient interactions, and potentially user-ingredient interactions.
# - Factorization Machine: In addition to the graph-based techniques, a Factorization Machines to model pairwise interactions between users, dishes, and ingredients. Factorization Machines allow one to capture non-linear interactions and latent relationships between entities.
class FactorizationMachineLayer(nn.Module):
    def __init__(self, reduce_sum: bool = True) -> None:
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x**2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return torch.Tensor(0.5 * ix)


class FactorizationMachineModel_withGCN(torch.nn.Module):
    def __init__(
        self,
        num_users: int,
        num_dishes: int,
        num_ingredients: int,
        embedding_dim: int,
        field_dims: List[int],
        features_dish_ingredient: torch.Tensor,
        features_user_dish: torch.Tensor,
        A_dish_ingredient: Optional[torch.Tensor] = None,
        A_user_dish: Optional[torch.Tensor] = None,
        cooccurrence_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(
            num_embeddings=num_users, embedding_dim=embedding_dim
        )
        self.dish_embeddings = nn.Embedding(
            num_embeddings=num_dishes, embedding_dim=embedding_dim
        )
        self.ingredient_embeddings = nn.Embedding(
            num_embeddings=num_ingredients, embedding_dim=embedding_dim
        )

        self.gce_dish_ingredient = GCELayerDishIngredient(
            field_dims=field_dims,
            embedding_dim=embedding_dim,
            features_dish_ingredient=features_dish_ingredient,
            A_dish_ingredient=A_dish_ingredient,
            attention=False,
            cooccurrence_weight=cooccurrence_weight,
        )
        self.gce_user_dish = GCELayerUserDish(
            field_dims=field_dims,
            embedding_dim=embedding_dim,
            features_user_dish=features_user_dish,
            A_user_dish=A_user_dish,
            attention=False,
            cooccurrence_weight=cooccurrence_weight,
        )

        self.projection_layer = nn.Linear(2 * embedding_dim, embedding_dim)

        self.fm = FactorizationMachineLayer(reduce_sum=True)

    def forward(
        self, users: torch.Tensor, dishes: torch.Tensor, ingredients: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate Graph Convolution Embeddings for dish-ingredient graph
        gce_dish_ingredient_embeddings = self.gce_dish_ingredient(
            torch.cat([dishes, ingredients], dim=1)
        )

        # Calculate Graph Convolution Embeddings for user-dish graph
        gce_user_dish_embeddings = self.gce_user_dish(torch.cat([users, dishes], dim=1))

        # Combine embeddings using the projection layer
        combined_embeddings = torch.cat(
            [gce_dish_ingredient_embeddings, gce_user_dish_embeddings], dim=1
        )
        combined_embeddings = torch.relu(self.projection_layer(combined_embeddings))

        # Calculate dot product between user-dish interactions and ingredient embeddings for FM
        pred_scores = torch.matmul(
            torch.mul(self.user_embeddings(users), self.dish_embeddings(dishes)),
            self.ingredient_embeddings(ingredients).weight.t(),
        )

        # Factorization Machine for combined embeddings
        fm_scores = self.fm(combined_embeddings)

        return pred_scores, fm_scores
