import os
import sys
from argparse import ArgumentParser, Namespace
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
from src.data.dish_ingredient_dataset import DishIngredientDataset
from src.data.ingredient_user_dataset import UserIngredientDataset


def build_adjacency_matrix(data: npt.NDArray[np.int32]) -> coo_matrix:
    user_ids = np.unique(data[:, 0])
    item_ids = np.unique(data[:, 1])

    user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_id_to_index = {item_id: index for index, item_id in enumerate(item_ids)}

    edges = []
    if data.shape[1] == 2:
        for user_id, item_id in data:
            edges.append((user_id_to_index[user_id], item_id_to_index[item_id]))
    elif data.shape[1] == 3:
        for user_id, item_id, _ in data:  # third element is the number of viewss
            edges.append((user_id_to_index[user_id], item_id_to_index[item_id]))
    else:
        raise ValueError(
            f"Data must have 2 or 3 columns, but got {data.shape[1]} columns."
        )

    adjacency_matrix = coo_matrix(
        (np.ones(len(edges)), zip(*edges)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.int32,
    )

    return adjacency_matrix


def negative_sampling(
    adj_matrix: coo_matrix, num_neg_samples: int = 4
) -> List[Tuple[int, int, int]]:
    # Get the indices of positive interactions (non-zero elements) from the adjacency matrix
    pos_indices = list(zip(adj_matrix.row, adj_matrix.col))

    num_users, num_items = adj_matrix.shape
    all_indices = set(range(num_users)), set(range(num_items))

    # Sample negative interactions randomly by excluding the positive interactions
    neg_samples = []
    for user, item in pos_indices:
        for _ in range(num_neg_samples):
            negative_item = np.random.choice(list(all_indices[1] - set([item])))
            neg_samples.append(
                (user, negative_item, 0)
            )  # Using 0 as the label for negative samples

    return neg_samples


def preprocessing(args: Namespace) -> None:
    """
    Preprocessing the data and save it as an artifact.
    """

    # Dish Ingredient Dataset
    dish_ingredient_df = pd.read_csv(args.data_path_dish_ingredient, sep=";")
    train_dish_ingredient, test_dish_ingredient = train_test_split(
        dish_ingredient_df.astype("int32").to_numpy(),
        train_size=args.train_size,
        random_state=args.random_state,
    )

    # Adjacency Matrix
    train_dish_ingredient_mat = build_adjacency_matrix(data=train_dish_ingredient)
    np.save(
        f"{args.output_path}/train_dish_ingredient_adjacency_matrix.npy",
        train_dish_ingredient_mat,
    )
    test_dish_ingredient_mat = build_adjacency_matrix(data=test_dish_ingredient)
    np.save(
        f"{args.output_path}/test_dish_ingredient_adjacency_matrix.npy",
        test_dish_ingredient_mat,
    )

    # Datasets
    dish_ingredient_train = DishIngredientDataset(
        data=train_dish_ingredient, dims=np.max(train_dish_ingredient, axis=0)
    )
    dish_ingredient_test = DishIngredientDataset(
        data=test_dish_ingredient, dims=np.max(test_dish_ingredient, axis=0)
    )
    torch.save(dish_ingredient_train, f"{args.output_path}/dish_ingredient_train.pt")
    torch.save(dish_ingredient_test, f"{args.output_path}/dish_ingredient_test.pt")

    # User Ingredient Dataset
    user_ingredient_df = pd.read_csv(args.data_path_ingredient_user, sep=";")
    train_user_ingredient, test_user_ingredient = train_test_split(
        user_ingredient_df.astype("int32").to_numpy(),
        train_size=args.train_size,
        random_state=args.random_state,
    )

    # Adjacency Matrix
    train_user_ingredient_mat = build_adjacency_matrix(data=train_user_ingredient)
    np.save(
        f"{args.output_path}/train_user_ingredient_adjacency_matrix.npy",
        train_user_ingredient_mat,
    )
    test_user_ingredient_mat = build_adjacency_matrix(data=test_user_ingredient)
    np.save(
        f"{args.output_path}/test_user_ingredient_adjacency_matrix.npy",
        test_user_ingredient_mat,
    )

    # Datasets
    user_ingredient_train = UserIngredientDataset(
        data=train_user_ingredient, dims=np.max(train_user_ingredient, axis=0)
    )
    user_ingredient_test = UserIngredientDataset(
        data=test_user_ingredient, dims=np.max(test_user_ingredient, axis=0)
    )
    torch.save(
        user_ingredient_train, f"{args.output_path}/user_ingredient_train.pt"
    )  # [499 (users), 14548 (ingredients), 10 (num_views))]
    torch.save(
        user_ingredient_test, f"{args.output_path}/user_ingredient_test.pt"
    )  # [499 (users), 14543 (ingredients), 10 (num_views))]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path_dish_ingredient",
        type=str,
        default=f"{os.getcwd()}/data/dish_ingredient.csv",
    )
    parser.add_argument(
        "--data_path_ingredient_user",
        type=str,
        default=f"{os.getcwd()}/data/user_ingredient.csv",
    )
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--output_path", type=str, default=f"{os.getcwd()}/data/preprocessed"
    )
    # Add additional arguments for the dimensions of users, dishes, and ingredients
    parser.add_argument("--num_users", type=int, default=499)
    parser.add_argument("--num_dishes", type=int, default=13501)
    parser.add_argument("--num_ingredients", type=int, default=14548)
    preprocessing(args=parser.parse_args())
