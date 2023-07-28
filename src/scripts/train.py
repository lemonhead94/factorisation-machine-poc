import os
import sys
from argparse import ArgumentParser, Namespace
from statistics import mean
from typing import Tuple

import torch
import torch.nn as nn
from sklearn.metrics import label_ranking_average_precision_score
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

sys.path.append(os.getcwd())
from src.model.factorisation_machine import FactorizationMachineModel_withGCN
from src.utils.metrics import getHitRatio, getNDCG


def test(
    model: nn.Module, test_x: Dataset, device: torch.device, topk: int = 10
) -> Tuple[float, float, float]:
    # Test the HR, NDCG, MPR for the model @topK
    model.eval()

    HR, NDCG, MPR = [], [], []
    for user_test in test_x:
        gt_item = user_test[0][1]
        predictions = model.predict(user_test, device)
        _, indices = torch.topk(predictions, topk)
        recommend_list = user_test[indices.cpu().detach().numpy()][:, 1]

        HR.append(getHitRatio(recommend_list, gt_item))
        NDCG.append(getNDCG(recommend_list, gt_item))
        MPR.append(label_ranking_average_precision_score(gt_item, recommend_list))
    return mean(HR), mean(NDCG), mean(MPR)


def train_one_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = []

    for _, (interactions, targets) in enumerate(data_loader):
        interactions = interactions.to(device)
        targets = targets.to(device)

        predictions = model(interactions)

        loss = criterion(predictions, targets.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    return mean(total_loss)


def train(args: Namespace) -> None:
    device = torch.device(args.device)

    # Load preprocessed datasets using
    user_ingredient_train = torch.load("user_ingredient_train.pt")
    dish_ingredient_train = torch.load("dish_ingredient_train.pt")
    user_ingredient_test = torch.load("user_ingredient_test.pt")

    fm_gcn = FactorizationMachineModel_withGCN(
        num_users=args.num_users,
        num_dishes=args.num_dishes,
        num_ingredients=args.num_ingredients,
        embedding_dim=args.embedding_dim,
        field_dims=3,
        features_dish_ingredient=dish_ingredient_train,
        features_user_ingredient=user_ingredient_train,
        cooccurrence_weight=2.0,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(params=fm_gcn.parameters(), lr=0.01)
    data_loader = DataLoader(
        user_ingredient_train, batch_size=args.batch_size, shuffle=True
    )

    # Run the training loop
    for epoch_i in trange(args.num_epochs):
        train_loss = train_one_epoch(
            model=fm_gcn,
            optimizer=optimizer,
            data_loader=data_loader,
            criterion=criterion,
            device=device,
        )
        hr, ndcg, mpr = test(fm_gcn, user_ingredient_test, device, topk=args.topk)

        print(f"Epoch: {epoch_i + 1}")
        print(f"train/loss: {train_loss}")
        print(f"eval/HR@{args.topk}: {hr}")
        print(f"eval/NDCG@{args.topk}: {ndcg}")
        print(f"eval/MPR@{args.topk}: {mpr}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--num_users", type=int, default=499)
    parser.add_argument("--num_dishes", type=int, default=13501)
    parser.add_argument("--num_ingredients", type=int, default=14548)
    parser.add_argument("--topk", type=int, default=5)
    train(args=parser.parse_args())
