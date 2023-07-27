import os
import sys

import torch
import torch.nn as nn
from sklearn.metrics import label_ranking_average_precision_score, ndcg_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.factorisation_machine import FactorizationMachineModel_withGCN

sys.path.append(os.getcwd())


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    num_batches = len(train_loader)

    for data in tqdm(train_loader, desc="Training..."):
        interaction_pairs, adjacency_matrix = data
        interaction_pairs = interaction_pairs.to(device)
        adjacency_matrix = adjacency_matrix.to(device)

        optimizer.zero_grad()
        outputs = model(interaction_pairs, adjacency_matrix)
        loss = criterion(outputs, target)  # Replace 'target' with your target variable
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / num_batches
    return avg_train_loss


def evaluate(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        # Lists to store true and predicted labels
        true_labels = []
        predicted_labels = []

        for data in tqdm(test_loader, desc="Evaluating..."):
            interaction_pairs, adjacency_matrix = data
            interaction_pairs = interaction_pairs.to(device)
            adjacency_matrix = adjacency_matrix.to(device)

            model(interaction_pairs, adjacency_matrix)

            # Your logic to convert the model outputs to top-k recommendations
            # For example, you can use torch.topk to get top-k indices
            # and convert them to binary labels indicating whether an ingredient is recommended or not.

            true_labels.append(...)  # Replace with the true binary labels
            predicted_labels.append(...)  # Replace with the predicted binary labels

        # Calculate NDCG and MPR
        ndcg = ndcg_score(true_labels, predicted_labels)
        mpr = label_ranking_average_precision_score(true_labels, predicted_labels)

    return ndcg, mpr


if __name__ == "__main__":
    # Set your hyperparameters and evaluation settings
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    top_k = 10  # Top-k recommendations for evaluation

    # Load preprocessed datasets
    user_ingredient_train = torch.load("user_ingredient_train.joblib")
    dish_ingredient_train = torch.load("dish_ingredient_train.joblib")
    user_ingredient_test = torch.load("user_ingredient_test.joblib")
    dish_ingredient_test = torch.load("dish_ingredient_test.joblib")

    # Create DataLoaders for training and testing
    user_ingredient_train_loader = DataLoader(
        user_ingredient_train, batch_size=batch_size, shuffle=True
    )
    dish_ingredient_train_loader = DataLoader(
        dish_ingredient_train, batch_size=batch_size, shuffle=True
    )
    user_ingredient_test_loader = DataLoader(
        user_ingredient_test, batch_size=batch_size, shuffle=False
    )
    dish_ingredient_test_loader = DataLoader(
        dish_ingredient_test, batch_size=batch_size, shuffle=False
    )

    # Initialize the model and optimizer
    #     def __init__(self, num_users, num_dishes, num_ingredients, embed_dim, field_dims, features_dish_ingredient, features_user_dish, A_dish_ingredient, A_user_dish, cooccurrence_weight=1.0) -> None:
    model = FactorizationMachineModel_withGCN(
        field_dims=[...], embed_dim=...
    )  # Replace with appropriate field_dims and embed_dim
    model_gcn_att = None
    gat_optimizer = torch.optim.Adam(params=model_gcn_att.parameters(), lr=0.01)

    # Define loss function
    criterion = (
        nn.BCEWithLogitsLoss()
    )  # Binary Cross-Entropy Loss for multi-label classification

    # Move model and data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        train_loss_user_ingredient = train(
            model, user_ingredient_train_loader, gat_optimizer, criterion, device
        )
        train_loss_dish_ingredient = train(
            model, dish_ingredient_train_loader, gat_optimizer, criterion, device
        )

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"User-Ingredient Training Loss: {train_loss_user_ingredient:.4f}")
        print(f"Dish-Ingredient Training Loss: {train_loss_dish_ingredient:.4f}")

        # Evaluation on test data
        ndcg_user, mpr_user = evaluate(model, user_ingredient_test_loader, device)
        ndcg_dish, mpr_dish = evaluate(model, dish_ingredient_test_loader, device)

        print(f"NDCG for User-Ingredient: {ndcg_user:.4f}")
        print(f"MPR for User-Ingredient: {mpr_user:.4f}")
        print(f"NDCG for Dish-Ingredient: {ndcg_dish:.4f}")
        print(f"MPR for Dish-Ingredient: {mpr_dish:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "trained_model.pt")
