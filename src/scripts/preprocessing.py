import os
import sys
from argparse import ArgumentParser, Namespace

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())
from src.data.dish_ingredient_dataset import DishIngredientDataset
from src.data.ingredient_user_dataset import IngredientUserDataset


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
    user_ingredient_train = IngredientUserDataset(data=train, dims=np.max(train, axis=0))
    user_ingredient_test = IngredientUserDataset(data=test, dims=np.max(test, axis=0))
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
