import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PairedTimeSet(Dataset):
    def __init__(self, X, Xp, U, indices):
        self.X = torch.from_numpy(X[indices]).float()
        self.Xp = torch.from_numpy(Xp[indices]).float()
        self.U = torch.from_numpy(U[indices]).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Xp[i], self.U[i]

class WindowedDataset(Dataset):
    def __init__(self, Xn, Un, traj_ids, T_roll, indices):
        self.Xn = torch.from_numpy(Xn).float()
        self.Un = torch.from_numpy(Un).float()
        self.traj_ids = traj_ids
        self.T = T_roll
        self.idxs = []

        N = Xn.shape[0]
        for k in indices:
            k = int(k)
            k_end = k + self.T
            if k_end >= N:
                continue
            if np.all(traj_ids[k:k_end+1] == traj_ids[k]):
                self.idxs.append(k)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        k = self.idxs[i]
        x0 = self.Xn[k]
        u_seq = self.Un[k:k+self.T]
        x_seq = self.Xn[k+1:k+self.T+1]

        return x0, u_seq, x_seq

def split_by_trajectory(traj_ids, train_ratio=0.6, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    unique_traj = np.unique(traj_ids)
    rng.shuffle(unique_traj)

    n_traj = len(unique_traj)
    n_train = int(train_ratio * n_traj)
    n_val = int(val_ratio * n_traj)

    train_traj = unique_traj[:n_train]
    val_traj = unique_traj[n_train:n_train+n_val]
    test_traj = unique_traj[n_train+n_val:]

    train_idx = np.where(np.isin(traj_ids, train_traj))[0]
    val_idx = np.where(np.isin(traj_ids, val_traj))[0]
    test_idx = np.where(np.isin(traj_ids, test_traj))[0]

    return train_idx, val_idx, test_idx

def compute_trim(Xk, Uk, vel_idx=(3,4,5), threshold=0.05):
    mask = np.linalg.norm(Xk[:, vel_idx], axis=1) < threshold
    if np.any(mask):
        x_trim = np.mean(Xk[mask], axis=0)
        u_trim = np.mean(Uk[mask], axis=0)
    else:
        x_trim = np.mean(Xk, axis=0)
        u_trim = np.mean(Uk, axis=0)

    return x_trim, u_trim


def normalize_data(Xk, Xkp1, Uk, train_idx):
    X_centered = Xk
    Xkp1_centered = Xkp1
    U_centered = Uk

    mu_x = X_centered[train_idx].mean(axis=0, keepdims=True)
    std_x = X_centered[train_idx].std(axis=0, keepdims=True) + 1e-8

    mu_u = U_centered[train_idx].mean(axis=0, keepdims=True)
    std_u = U_centered[train_idx].std(axis=0, keepdims=True) + 1e-8

    Xk_n = (X_centered - mu_x) / std_x
    Xkp1_n = (Xkp1_centered - mu_x) / std_x
    Uk_n = (U_centered - mu_u) / std_u

    stats = {"mu_x": mu_x,
            "std_x": std_x,
            "mu_u": mu_u,
            "std_u": std_u}
    return Xk_n, Xkp1_n, Uk_n, stats

def build_dataloaders(Xk, Xkp1, Uk, traj_ids, batch_size=1024, T_roll=10, train_ratio=0.6, val_ratio=0.2, seed=0):
    train_idx, val_idx, test_idx = split_by_trajectory(traj_ids, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    x_trim, u_trim = compute_trim(Xk, Uk)

    X_centered = Xk - x_trim
    Xkp1_centered = Xkp1 - x_trim
    U_centered = Uk - u_trim

    Xk_n, Xkp1_n, Uk_n, stats = normalize_data(X_centered, Xkp1_centered, U_centered,train_idx)

    train_ds = PairedTimeSet(Xk_n, Xkp1_n, Uk_n, train_idx)
    val_ds = PairedTimeSet(Xk_n, Xkp1_n, Uk_n, val_idx)
    test_ds = PairedTimeSet(Xk_n, Xkp1_n, Uk_n, test_idx)

    train_ds_roll = WindowedDataset(Xk_n, Uk_n, traj_ids, T_roll, train_idx)
    val_ds_roll = WindowedDataset(Xk_n, Uk_n, traj_ids, T_roll, val_idx)
    test_ds_roll = WindowedDataset(Xk_n, Uk_n, traj_ids, T_roll, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    train_loader_roll = DataLoader(train_ds_roll, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader_roll = DataLoader(val_ds_roll, batch_size=batch_size, shuffle=False)
    test_loader_roll = DataLoader(test_ds_roll, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_loader_roll": train_loader_roll,
        "val_loader_roll": val_loader_roll,
        "test_loader_roll": test_loader_roll,
        "stats": stats,
        "trim": (x_trim, u_trim),
        "splits": (train_idx, val_idx, test_idx)}