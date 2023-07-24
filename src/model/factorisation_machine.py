import torch
import torch.nn as nn

from src.model.gce_layer import GCELayerDishIngredient, GCELayerUserDish


# Graph-Aware Recommender Systems with Heterogeneous Graph Fusion for Enhanced Ingredient Recommendations
# - Graph-Aware Recommender System: a RS that incorporates graph-based techniques to model user-item interactions and item-item relationships. Interactions involve users, dishes, and ingredients, forming a heterogeneous information network.
# - Heterogeneous Graph Fusion: multiple graphs or different types of interactions in the heterogeneous information network. These interactions include user-dish interactions, dish-ingredient interactions, and potentially user-ingredient interactions.
# - Factorization Machine: In addition to the graph-based techniques, a Factorization Machines to model pairwise interactions between users, dishes, and ingredients. Factorization Machines allow one to capture non-linear interactions and latent relationships between entities.
class FactorizationMachineLayer(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

class FactorizationMachineModel_withGCN(torch.nn.Module):
    def __init__(self, num_users, num_dishes, num_ingredients, embed_dim, field_dims, features_dish_ingredient, features_user_dish, A_dish_ingredient, A_user_dish, cooccurrence_weight=1.0):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embed_dim)
        self.dish_embeddings = nn.Embedding(num_dishes, embed_dim)
        self.ingredient_embeddings = nn.Embedding(num_ingredients, embed_dim)

        self.gce_dish_ingredient = GCELayerDishIngredient(field_dims, embed_dim, features_dish_ingredient, A_dish_ingredient, cooccurrence_weight=cooccurrence_weight)
        self.gce_user_dish = GCELayerUserDish(field_dims, embed_dim, features_user_dish, A_user_dish, cooccurrence_weight=cooccurrence_weight)

        self.projection_layer = nn.Linear(2 * embed_dim, embed_dim)

        self.fm = FactorizationMachineLayer(reduce_sum=True)

    def forward(self, users, dishes, ingredients):
        user_embeds = self.user_embeddings(users)
        dish_embeds = self.dish_embeddings(dishes)
        ingredient_embeds = self.ingredient_embeddings(ingredients)

        # Calculate Graph Convolution Embeddings for dish-ingredient graph
        gce_dish_ingredient_embeds = self.gce_dish_ingredient(torch.cat([dishes, ingredients], dim=1))

        # Calculate Graph Convolution Embeddings for user-dish graph
        gce_user_dish_embeds = self.gce_user_dish(torch.cat([users, dishes], dim=1))

        # Combine embeddings using the projection layer
        combined_embeds = torch.cat([gce_dish_ingredient_embeds, gce_user_dish_embeds], dim=1)
        combined_embeds = torch.relu(self.projection_layer(combined_embeds))

        # Calculate dot product between user-dish interactions and ingredient embeddings for FM
        pred_scores = torch.matmul(torch.mul(user_embeds, dish_embeds), ingredient_embeds.t())

        # Factorization Machine for combined embeddings
        fm_scores = self.fm(combined_embeds)

        return pred_scores, fm_scores
