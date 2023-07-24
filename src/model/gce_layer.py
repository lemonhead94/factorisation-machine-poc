import torch


class BaseGCELayer(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, features, A, attention=False, cooccurrence_weight=1.0):
        super().__init__()
        self.A = A
        self.features = features
        self.cooccurrence_weight = cooccurrence_weight

        if attention:
            self.GCN_module = GATConv(int(field_dims), embed_dim, heads=8, dropout=0.4)
        else:
            self.GCN_module = GCNConv(field_dims, embed_dim)

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in derived classes.")


class GCELayerDishIngredient(BaseGCELayer):
    def __init__(self, field_dims, embed_dim, features_dish_ingredient, A_dish_ingredient, attention=False, cooccurrence_weight=1.0):
        super().__init__(field_dims, embed_dim, features_dish_ingredient, A_dish_ingredient, attention=attention, cooccurrence_weight=cooccurrence_weight)

    def forward(self, x):
        # Custom forward pass for dish-ingredient interactions
        gce_embeddings = self.GCN_module(self.features, self.A)[x]
        user_ingredient_embeddings = torch.mul(gce_embeddings, self.cooccurrence_weight)
        return user_ingredient_embeddings

class GCELayerUserDish(BaseGCELayer):
    def __init__(self, field_dims, embed_dim, features_user_dish, A_user_dish, attention=False, cooccurrence_weight=1.0):
        super().__init__(field_dims, embed_dim, features_user_dish, A_user_dish, attention=attention, cooccurrence_weight=cooccurrence_weight)

    def forward(self, x):
        # Custom forward pass for user-dish interactions
        gce_embeddings = self.GCN_module(self.features, self.A)[x]
        return gce_embeddings
