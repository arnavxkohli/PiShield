import argparse
import logging
import random
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from pishield.linear_requirements.classes import Constraint
from pishield.linear_requirements.parser import parse_constraints_file, remap_constraint_variables
from pishield.linear_requirements.shield_layer import ShieldLayer
from pishield.linear_requirements.ste import STEShield
from pishield.linear_requirements.kkt_ste import KKTShieldSTE, build_constraint_matrix


def setup_logging(log_file: str):
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger()
    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(precedence=('cuda', 'mps', 'cpu')) -> torch.device:
    device = torch.device('cpu')
    for dev in precedence:
        if dev == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        if dev == 'mps' and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            device = torch.device('mps')
    logging.info(f"Using: {device}") 
    return device


def load_data(
    csv_path: str,
    constraints_file: str,
    test_size: float,
    val_size: float,
    seed: int,
    scale_targets: bool = True
):
    df = pd.read_csv(csv_path)
    # parse constraints
    ordering, constraints = parse_constraints_file(constraints_file)
    ordering, constraints, rev_map = remap_constraint_variables(ordering, constraints)
    # identify target columns
    target_cols = [df.columns[int(var.split('_')[1])] for var in rev_map.values()]
    features = [c for c in df.columns if c not in target_cols]

    # map status labels
    if 'status' in df and df['status'].dtype == object:
        df['status'] = df['status'].map({'legitimate': 0, 'phishing': 1})

    X = df[features]
    y = df[target_cols]

    # split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=seed
    )

    # scale continuous features
    cont_feats = [c for c in features if df[c].nunique() > 2]
    cat_feats = [c for c in features if c not in cont_feats]
    preproc = ColumnTransformer(
        [('num', StandardScaler(), cont_feats), ('cat', 'passthrough', cat_feats)]
    )
    if scale_targets:
        target_scaler = StandardScaler()
    else:
        target_scaler = None

    X_train_t = torch.tensor(preproc.fit_transform(X_train), dtype=torch.float32)
    X_val_t = torch.tensor(preproc.transform(X_val), dtype=torch.float32)
    X_test_t = torch.tensor(preproc.transform(X_test), dtype=torch.float32)

    if scale_targets:
        y_train_t = torch.tensor(target_scaler.fit_transform(y_train.values), dtype=torch.float32)
        y_val_t = torch.tensor(target_scaler.transform(y_val.values), dtype=torch.float32)
        y_test_t = torch.tensor(target_scaler.transform(y_test.values), dtype=torch.float32)
    else:
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
        y_val_t = torch.tensor(y_val.values, dtype=torch.float32)
        y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

    return (
        (X_train_t, y_train_t),
        (X_val_t, y_val_t),
        (X_test_t, y_test_t),
        constraints,
        len(features),
        len(target_cols),
        target_scaler
    )


def make_dataloaders(data, batch_size: int):
    loaders = {}
    for name, (X, y) in zip(['train', 'val', 'test'], data):
        shuffle = (name == 'train')
        loaders[name] = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)
    return loaders


def validate_constraints(constraints: list[Constraint], preds: torch.Tensor, logger: logging.Logger, verbose=True) -> bool:
    if verbose:
        logger.info("Starting Constraint Check...")
    all_ok = True
    for c in constraints:
        if not c.check_satisfaction(preds):
            if verbose:
                logger.warning(f"Constraint violated: {c.readable()}")
            all_ok = False
    if verbose:
        logger.info("Constraint Check Complete!")
    return all_ok


class BasicMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)


class ShieldedMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, constraints_file: str):
        super().__init__()
        self.mlp = BasicMLP(input_size, output_size)
        self.shield = ShieldLayer(output_size, constraints_file)

    def forward(self, x):
        preds = self.mlp(x)
        return self.shield(preds)


class ShieldedMLPWithSTE(nn.Module):
    def __init__(self, input_size: int, output_size: int, constraints_file: str):
        super().__init__()
        self.mlp = BasicMLP(input_size, output_size)
        self.shield = ShieldLayer(output_size, constraints_file)

    def forward(self, x):
        preds = self.mlp(x)
        return STEShield.apply(preds, self.shield)


class ShieldedMLPWithKKTSTE(nn.Module):
    def __init__(self, input_size: int, output_size: int, constraints_file: str):
        super().__init__()
        self.mlp = BasicMLP(input_size, output_size)
        self.shield = ShieldLayer(output_size, constraints_file)

        A, b = build_constraint_matrix(self.shield.constraints, output_size)
        self.register_buffer("A", A)
        self.register_buffer("b", b)

    def forward(self, x):
        preds = self.mlp(x)
        return KKTShieldSTE.apply(preds, self.A.to(preds.device), self.b.to(preds.device), self.shield)



def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    constraints: list[Constraint],
    device: torch.device,
    logger: logging.Logger,
    verbose=True
) -> tuple[float, int, int]:
    model.train()
    total_loss = 0.0
    runs, sat = 0, 0

    for X_batch, y_batch in tqdm(loader, desc='Training', leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = F.mse_loss(outputs, y_batch)

        sat += validate_constraints(constraints, outputs, logger, verbose)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        runs += 1

    return total_loss / runs, sat, runs


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    constraints: list[Constraint],
    target_scaler: StandardScaler,
    device: torch.device,
    logger: logging.Logger,
    verbose: bool = True
) -> tuple[float, int, int]:
    model.eval()
    preds_list, targets_list = [], []
    runs, sat = 0, 0

    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc='Evaluating', leave=False):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)

            preds_list.append(outputs.cpu())
            targets_list.append(y_batch.cpu())

            sat += validate_constraints(constraints, outputs, logger, verbose)
            runs += 1

    all_preds = torch.cat(preds_list)  # shape (N, D)
    all_targets = torch.cat(targets_list)

    if target_scaler is not None:
        all_preds = torch.tensor(target_scaler.inverse_transform(all_preds.numpy()), dtype=torch.float32)
        all_targets = torch.tensor(target_scaler.inverse_transform(all_targets.numpy()), dtype=torch.float32)

    rmse = torch.sqrt(F.mse_loss(all_preds, all_targets)).item()

    return rmse, sat, runs


def main(args):
    logger = setup_logging(os.path.join(args.data_dir, 'training_log.txt'))
    logger.info('Starting experiment')
    set_seed(args.seed)
    device = get_device()

    logger.info(f'Seed={args.seed}, Device={device}, Epochs={args.epochs}, Batch={args.batch_size}')
    constraints_file = os.path.join(args.data_dir, 'constraints.txt')
    data_file = os.path.join(args.data_dir, 'data.csv')

    data = load_data(
        data_file,
        constraints_file,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        scale_targets=args.scale_targets
    )
    (train_data, val_data, test_data, constraints, in_dim, out_dim, target_scaler) = data
    logger.info(f"Constraints satisfied on training data: {validate_constraints(constraints, train_data[1], logger, verbose=False)}")
    logger.info(f"Constraints satisfied on validation data: {validate_constraints(constraints, val_data[1], logger, verbose=False)}")
    logger.info(f"Constraints satisfied on test data: {validate_constraints(constraints, test_data[1], logger, verbose=False)}")
    loaders = make_dataloaders((train_data, val_data, test_data), args.batch_size)

    overall_start_time = time.time()

    for desc, model_ctor in [
        ('BasicMLP', lambda: BasicMLP(in_dim, out_dim)),
        ('ShieldNoLoss', lambda: ShieldedMLP(in_dim, out_dim, constraints_file)),
        ('ShieldWithKKTSTE', lambda: ShieldedMLPWithKKTSTE(in_dim, out_dim, constraints_file))
    ]:
        logger.info(f'== Experiment: {desc} ==')
        model = model_ctor().to(device)
        opt = optim.Adam(model.parameters(), lr=args.lr)

        model_start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            train_loss, train_sat, train_runs = train_epoch(
                model, loaders['train'], opt, constraints, device, logger, verbose=False
            )
            val_rmse, val_sat, val_runs = eval_epoch(
                model, loaders['val'], constraints, target_scaler, device, logger, verbose=False
            )

            epoch_time = time.time() - epoch_start_time
            result_string = f'{desc} Epoch {epoch}/{args.epochs} ' \
                f'Loss={train_loss:.4f} TrainSat={train_sat}/{train_runs} ' \
                f'ValRMSE={val_rmse:.4f} ValSat={val_sat}/{val_runs} ' \
                f'EpochTime={epoch_time:.2f}s'
            print(f"\n{result_string}")
            logger.info(result_string)

        # test
        test_start_time = time.time()
        test_rmse, test_sat, test_runs = eval_epoch(
            model, loaders['test'], constraints, target_scaler, device, logger, verbose=False
        )
        test_time = time.time() - test_start_time
        total_model_time = time.time() - model_start_time

        result_string = f'{desc} TestRMSE={test_rmse:.4f} Sat={test_sat}/{test_runs} ' \
                        f'TestTime={test_time:.2f}s TotalTrainTime={total_model_time:.2f}s'
        print(f"{result_string}\n")
        logger.info(result_string)

    logger.info(f'Overall time: {(time.time() - overall_start_time):.2f}s')
    logger.info('Experiment finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train shield models')
    parser.add_argument('--data-dir', default='faulty-steel-plates')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--val-size', type=float, default=0.125)
    parser.add_argument('--scale-targets', type=bool, default=False)
    args = parser.parse_args()
    main(args)
