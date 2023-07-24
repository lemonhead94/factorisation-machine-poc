import os
import sys
from argparse import ArgumentParser, Namespace

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.data.dish_ingredient_dataset import DishIngredientDataset
from src.data.ingredient_user_dataset import UserIngredientDataset


def build_adj_mx(n_feat, data) -> sp.dok_matrix:
    train_mat = sp.dok_matrix((n_feat, n_feat), dtype=np.float32)
    for x in tqdm(data, desc="Building Adjacency Matrix..."):
        train_mat[x[0], x[1]] = 1.0
        train_mat[x[1], x[0]] = 1.0
        if data.shape[1] > 2:
            for idx in range(len(x[2:])):
                train_mat[x[0], x[2 + idx]] = 1.0
                train_mat[x[1], x[2 + idx]] = 1.0
                train_mat[x[2 + idx], x[0]] = 1.0
                train_mat[x[2 + idx], x[1]] = 1.0
    return train_mat

def build_heterogeneous_adj_mx(dims, data_dish_ingredient, data_user_ingredient) -> sp.dok_matrix:
    num_users = dims[0]
    num_dishes = dims[1]
    dims[2]

    user_dish_mat = build_adj_mx(num_users, data_user_ingredient)
    dish_ingredient_mat = build_adj_mx(num_dishes, data_dish_ingredient)
    user_ingredient_mat = user_dish_mat @ dish_ingredient_mat.transpose()

    # Concatenate the three adjacency matrices to form the heterogeneous adjacency matrix
    # The order of concatenation matters. You can choose the order based on the importance of different interactions in your model.
    # In this example, we are assuming the order: user-dish, dish-ingredient, user-ingredient
    heterogeneous_adj_mx = sp.hstack([user_dish_mat, dish_ingredient_mat, user_ingredient_mat])
    return heterogeneous_adj_mx

def preprocessing(args: Namespace) -> None:
    """
    Preprocessing the data and save it as an artifact.
    """

    # Dish Ingredient Dataset
    dish_ingredient_df = pd.read_csv(args.data_path_dish_ingredient, sep=";")
    train_dish_ingredient, test_dish_ingredient = train_test_split(dish_ingredient_df.astype('int32').to_numpy(), train_size=args.train_size, random_state=args.random_state)
    dish_ingredient_train = DishIngredientDataset(data=train_dish_ingredient, dims=np.max(train_dish_ingredient, axis=0))
    dish_ingredient_test = DishIngredientDataset(data=test_dish_ingredient, dims=np.max(test_dish_ingredient, axis=0))
    joblib.dump(dish_ingredient_train, f"{args.output_path}/dish_ingredient_train.joblib")
    joblib.dump(dish_ingredient_test, f"{args.output_path}/dish_ingredient_test.joblib")

    # User Ingredient Dataset
    user_ingredient_df = pd.read_csv(args.data_path_ingredient_user, sep=";")
    train_user_ingredient, test_user_ingredient = train_test_split(user_ingredient_df.astype('int32').to_numpy(), train_size=args.train_size, random_state=args.random_state)
    user_ingredient_train = UserIngredientDataset(data=train_user_ingredient, dims=np.max(train_user_ingredient, axis=0))
    user_ingredient_test = UserIngredientDataset(data=test_user_ingredient, dims=np.max(test_user_ingredient, axis=0))
    joblib.dump(user_ingredient_train, f"{args.output_path}/user_ingredient_train.joblib") # [499 (users), 14548 (ingredients), 10 (num_views))]
    joblib.dump(user_ingredient_test, f"{args.output_path}/user_ingredient_test.joblib") # [499 (users), 14543 (ingredients), 10 (num_views))]

    # Heterogeneous Graph Fusion
    heterogeneous_adj_mx = build_heterogeneous_adj_mx(dims=[args.num_users, args.num_dishes, args.num_ingredients],
                                                     data_dish_ingredient=train_dish_ingredient,
                                                     data_user_ingredient=train_user_ingredient)
    sp.save_npz(f"{args.output_path}/heterogeneous_adjacency_matrix.npz", heterogeneous_adj_mx)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path_dish_ingredient", type=str, default=f"{os.getcwd()}/data/dish_ingredient.csv")
    parser.add_argument("--data_path_ingredient_user", type=str, default=f"{os.getcwd()}/data/user_ingredient.csv")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output_path", type=str, default=f"{os.getcwd()}/data/preprocessed")
    # Add additional arguments for the dimensions of users, dishes, and ingredients
    parser.add_argument("--num_users", type=int, default=499)
    parser.add_argument("--num_dishes", type=int, default=13501)
    parser.add_argument("--num_ingredients", type=int, default=14548)
    preprocessing(args=parser.parse_args())
