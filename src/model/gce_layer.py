import torch
from torch_geometric.nn import GATConv, GCNConv


class BaseGCELayer(torch.nn.Module):
    def __init__(
        self,
        field_dims: int,
        embedding_dim: int,
        features: torch.Tensor,
        attention_vec: torch.Tensor,
        attention: bool = False,
        attention_dropout: float = 0.4,
        num_multi_head_attentions: int = 8,
        cooccurrence_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.attention_vec = attention_vec
        self.features = features
        self.cooccurrence_weight = cooccurrence_weight

        if attention:
            self.graph_conv = GATConv(
                in_channels=field_dims,
                out_channels=embedding_dim,
                heads=num_multi_head_attentions,
                dropout=attention_dropout,
            )
        else:
            self.graph_conv = GCNConv(
                in_channels=field_dims, out_channels=embedding_dim
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Forward method must be implemented in derived classes."
        )


class GCELayerDishIngredient(BaseGCELayer):
    def __init__(
        self,
        field_dims: int,
        embedding_dim: int,
        features_dish_ingredient: torch.Tensor,
        attention_vec_dish_ingredient: torch.Tensor,
        attention: bool = False,
        cooccurrence_weight: float = 1.0,
    ) -> None:
        super().__init__(
            field_dims,
            embedding_dim,
            features_dish_ingredient,
            attention_vec_dish_ingredient,
            attention=attention,
            cooccurrence_weight=cooccurrence_weight,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Custom forward pass for dish-ingredient interactions
        gce_embeddings = self.graph_conv(self.features, self.attention_vec)[x]
        user_ingredient_embeddings = torch.mul(gce_embeddings, self.cooccurrence_weight)
        return user_ingredient_embeddings


class GCELayerUserDish(BaseGCELayer):
    def __init__(
        self,
        field_dims: int,
        embedding_dim: int,
        features_user_dish: torch.Tensor,
        attention_vec_user_dish: torch.Tensor,
        attention: bool = False,
        cooccurrence_weight: float = 1.0,
    ) -> None:
        super().__init__(
            field_dims,
            embedding_dim,
            features_user_dish,
            attention_vec_user_dish,
            attention=attention,
            cooccurrence_weight=cooccurrence_weight,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Custom forward pass for user-dish interactions
        gce_embeddings = self.graph_conv(self.features, self.attention_vec)[x]
        return torch.Tensor(gce_embeddings)
