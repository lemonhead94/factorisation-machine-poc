from typing import Optional

import torch
from torch_geometric.nn import GATConv, GCNConv


class BaseGCELayer(torch.nn.Module):
    def __init__(
        self,
        field_dims: int,
        embedding_dim: int,
        features: torch.Tensor,
        cooccurrence_weight: float,
        attention_vec: Optional[torch.Tensor] = None,
        attention: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.4,
        num_multi_head_attentions: Optional[int] = 8,
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
        cooccurrence_weight: float = 1.0,
    ) -> None:
        super().__init__(
            field_dims=field_dims,
            embedding_dim=embedding_dim,
            features=features_dish_ingredient,
            cooccurrence_weight=cooccurrence_weight,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Custom forward pass for dish-ingredient interactions
        gce_embeddings = self.graph_conv(self.features)[x]
        user_ingredient_embeddings = torch.mul(gce_embeddings, self.cooccurrence_weight)
        return user_ingredient_embeddings


class GCELayerUserIngredient(BaseGCELayer):
    def __init__(
        self,
        field_dims: int,
        embedding_dim: int,
        features_user_ingredient: torch.Tensor,
        cooccurrence_weight: float = 1.0,
    ) -> None:
        super().__init__(
            field_dims=field_dims,
            embedding_dim=embedding_dim,
            features=features_user_ingredient,
            cooccurrence_weight=cooccurrence_weight,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Custom forward pass for user-dish interactions
        gce_embeddings = self.graph_conv(self.features)[x]
        return torch.Tensor(gce_embeddings)
