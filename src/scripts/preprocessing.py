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

# https://arxiv.org/pdf/2005.09863.pdf
def ng_sample(data, dims, num_ng=4):
    rating_mat = build_adj_mx(dims[-1], data)
    interactions = []
    min_item, max_item = dims[0], dims[1]
    for num, x in tqdm(enumerate(data), desc='perform negative sampling...'):
        interactions.append(np.append(x, 1))
        for t in range(num_ng):
            j = np.random.randint(min_item, max_item)
            while (x[0], j) in rating_mat or j == int(x[1]):
                j = np.random.randint(min_item, max_item)
            interactions.append(np.concatenate([[x[0], j], x[2:], [0]]))
    return np.vstack(interactions), rating_mat

def preprocessing(args: Namespace) -> None:
    """
    Preprocessing the data and save it as an artifact.
    """

    # Dish Ingredient Dataset
    dish_ingredient_df = pd.read_csv(args.data_path_dish_ingredient, sep=";")
    train, test = train_test_split(dish_ingredient_df.astype('int32').to_numpy(), train_size=args.train_size, random_state=args.random_state)
    dish_ingredient_train = DishIngredientDataset(data=train, dims=np.max(train, axis=0)) # [13501 (dishes), 14552 (ingredients)]
    dish_ingredient_test = DishIngredientDataset(data=test, dims=np.max(test, axis=0)) # [13501 (dishes), 14548 (ingredients)]
    joblib.dump(dish_ingredient_train, f"{args.output_path}/dish_ingredient_train.joblib")
    joblib.dump(dish_ingredient_test, f"{args.output_path}/dish_ingredient_test.joblib")

    # User Ingredient Dataset
    user_ingredient_df = pd.read_csv(args.data_path_ingredient_user, sep=";")
    train, test = train_test_split(user_ingredient_df.astype('int32').to_numpy(), train_size=args.train_size, random_state=args.random_state)
    user_ingredient_train = UserIngredientDataset(data=train, dims=np.max(train, axis=0))
    user_ingredient_test = UserIngredientDataset(data=test, dims=np.max(test, axis=0))
    joblib.dump(user_ingredient_train, f"{args.output_path}/user_ingredient_train.joblib") # [499 (users), 14548 (ingredients), 10 (num_views))]
    joblib.dump(user_ingredient_test, f"{args.output_path}/user_ingredient_test.joblib") # [499 (users), 14543 (ingredients), 10 (num_views))]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path_dish_ingredient", type=str, default=f"{os.getcwd()}/data/dish_ingredient.csv")
    parser.add_argument("--data_path_ingredient_user", type=str, default=f"{os.getcwd()}/data/user_ingredient.csv")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output_path", type=str, default=f"{os.getcwd()}/data/preprocessed")
    preprocessing(args=parser.parse_args())
