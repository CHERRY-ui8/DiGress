# src/datasets/metabolite_dataset.py

import os
import os.path as osp
import pathlib

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import subgraph
from tqdm import tqdm

import src.utils as utils
from src.analysis.rdkit_functions import (
    build_molecule_with_partial_charges,
    compute_molecular_metrics,
    mol2smiles,
)
from src.datasets.abstract_dataset import AbstractDatasetInfos, MolecularDataModule

# ══════════════════════════════════════════════════════════════════
#  原子 / 化学键词表
#  覆盖 NIST20 + MoNA 中出现的全部重原子；
#  未在词表中的原子类型的分子将在 process() 阶段被过滤并记录。
# ══════════════════════════════════════════════════════════════════

# 不含氢版本（最常用）
ATOM_TYPES_NO_H = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
# 含氢版本
ATOM_TYPES_H    = ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']

# 最大形式化合价（用于 DiGress 的化学有效性检查）
# 使用较宽松的上限以覆盖真实代谢物（如 -SO3H, -PO4H2 等）
VALENCIES_NO_H = [4, 3, 2, 6, 5, 1, 1, 1, 1]   # C N O S P F Cl Br I
VALENCIES_H    = [1, 4, 3, 2, 6, 5, 1, 1, 1, 1] # H C N O S P F Cl Br I

# 原子量（用于分子量估算）
ATOM_WEIGHTS_NO_H = {0: 12, 1: 14, 2: 16, 3: 32, 4: 31, 5: 19, 6: 35, 7: 80,  8: 127}
ATOM_WEIGHTS_H    = {0:  1, 1: 12, 2: 14, 3: 16, 4: 32, 5: 31, 6: 19, 7: 35, 8: 80,  9: 127}

# 化学键类型：索引 0 保留给"无边"，1-4 对应四种化学键
BOND_TYPES = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

# 数据集划分比例与文件名（存储在 raw_dir 中保证可复现）
SPLIT_RATIOS = (0.8, 0.1, 0.1)
SPLIT_FILES  = ['split_train.npy', 'split_val.npy', 'split_test.npy']


# ══════════════════════════════════════════════════════════════════
#  Transform 类（与 QM9 对应）
# ══════════════════════════════════════════════════════════════════

class RemoveYTransform:
    """清空 y，用于无条件生成阶段"""
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class KeepDreaMSTransform:
    """保留 DreaMS embedding（data.y）用于条件生成"""
    def __call__(self, data):
        return data  # y 已在 process() 中写入，不做任何修改


# ══════════════════════════════════════════════════════════════════
#  MetaboliteDataset
# ══════════════════════════════════════════════════════════════════

class MetaboliteDataset(InMemoryDataset):
    """
    NIST20 / MoNA 代谢组数据集，适配 DiGress 的 InMemoryDataset 接口。

    数据来源（HDF5）：
        smiles           : (N,)        SMILES 字符串
        DreaMS_embedding : (N, 1024)   质谱 embedding，作为生成条件写入 data.y
        precursor_mz     : (N,)        前体离子 m/z，写入 data.precursor_mz
        id               : (N,)        分子 ID

    参数：
        stage      : 'train' | 'val' | 'test'
        root       : 缓存处理结果的根目录
        hdf5_path  : HDF5 文件绝对或相对路径
        remove_h   : True 则去掉氢原子（推荐），False 则保留显式氢
    """

    def __init__(self, stage: str, root: str, hdf5_path: str,
                 remove_h: bool = True,
                 transform=None, pre_transform=None, pre_filter=None):
        self.hdf5_path = osp.abspath(hdf5_path)
        self.remove_h  = remove_h
        self.stage     = stage
        self.file_idx  = {'train': 0, 'val': 1, 'test': 2}[stage]

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx],
                                            weights_only=False)

    # ── 文件名约定 ────────────────────────────────────────────────────────
    @property
    def raw_file_names(self):
        """InMemoryDataset 用这些文件是否存在来判断是否需要调用 download()"""
        return SPLIT_FILES

    @property
    def processed_file_names(self):
        suffix = 'no_h' if self.remove_h else 'h'
        return [f'proc_tr_{suffix}.pt',
                f'proc_val_{suffix}.pt',
                f'proc_test_{suffix}.pt']

    # ── 阶段一：划分 train/val/test 索引（仅运行一次）────────────────────
    def download(self):
        """
        不需要联网；只负责将 N 个样本随机划分为三折并保存索引。
        索引文件存储在 raw_dir/ 中以保证后续可复现。
        """
        split_paths = [osp.join(self.raw_dir, f) for f in SPLIT_FILES]
        if all(osp.exists(p) for p in split_paths):
            return   # 已经划分过，直接跳过

        with h5py.File(self.hdf5_path, 'r') as f:
            n_total = int(f['smiles'].shape[0])

        rng     = np.random.default_rng(seed=42)
        indices = rng.permutation(n_total)

        n_train = int(SPLIT_RATIOS[0] * n_total)
        n_val   = int(SPLIT_RATIOS[1] * n_total)
        splits  = np.split(indices, [n_train, n_train + n_val])

        for path, idx_arr in zip(split_paths, splits):
            np.save(path, idx_arr)

        print(f"[MetaboliteDataset] Dataset split completed | "
              f"train={len(splits[0])}, val={len(splits[1])}, test={len(splits[2])}")

    # ── 阶段二：SMILES → PyG Data（每个 stage 写自己的 .pt 文件）────────
    def process(self):
        RDLogger.DisableLog('rdApp.*')

        # 原子词表
        atom_list     = ATOM_TYPES_NO_H if self.remove_h else ATOM_TYPES_H
        types         = {sym: i for i, sym in enumerate(atom_list)}
        n_atom_types  = len(atom_list)
        n_bond_types  = len(BOND_TYPES)   # 4；加上"无边"共 5 类

        # 读取当前 stage 对应的全局索引
        split_idx = np.load(self.raw_paths[self.file_idx])

        # 一次性读入 HDF5（全量 load 进内存；79300 条数据量可接受）
        with h5py.File(self.hdf5_path, 'r') as f:
            all_smiles     = f['smiles'][:]               # bytes array
            all_embeddings = f['DreaMS_embedding'][:]     # (N, 1024) float32
            all_precursor  = f['precursor_mz'][:]         # (N,)      float64

        data_list  = []
        n_invalid  = 0   # RDKit 无法解析
        n_filtered = 0   # 含词表外原子/键

        for global_idx in tqdm(split_idx, desc=f'[{self.stage}] SMILES → Graph'):
            # ── 解码 SMILES ────────────────────────────────────────────
            raw  = all_smiles[global_idx]
            smi  = raw.decode() if isinstance(raw, bytes) else str(raw)

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                n_invalid += 1
                continue

            # ── 氢处理 ─────────────────────────────────────────────────
            mol = Chem.RemoveHs(mol) if self.remove_h else Chem.AddHs(mol)
            N   = mol.GetNumAtoms()

            # ── 节点：检查词表覆盖并收集类型索引 ─────────────────────
            type_idx = []
            valid    = True
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                if sym not in types:
                    valid = False
                    break
                type_idx.append(types[sym])

            if not valid:
                n_filtered += 1
                continue

            # ── 边：检查键类型并构建 COO 格式 ────────────────────────
            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bt   = bond.GetBondType()
                if bt not in BOND_TYPES:
                    valid = False
                    break
                row       += [s, e]
                col       += [e, s]
                # +1：为"无边"类型（索引 0）腾出位置，与 QM9 约定一致
                edge_type += 2 * [BOND_TYPES[bt] + 1]

            if not valid:
                n_filtered += 1
                continue

            # ── 组装张量 ───────────────────────────────────────────────
            if len(row) > 0:
                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_attr  = F.one_hot(
                    torch.tensor(edge_type, dtype=torch.long),
                    num_classes=n_bond_types + 1
                ).float()
                # 按首节点排序（保证确定性，与 QM9 一致）
                perm       = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr  = edge_attr[perm]
            else:
                # 单原子分子（极少数）
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr  = torch.zeros((0, n_bond_types + 1), dtype=torch.float)

            x = F.one_hot(torch.tensor(type_idx), num_classes=n_atom_types).float()

            # ── 条件特征 y：DreaMS embedding，shape=(1, 1024) ──────────
            # 与 QM9 的 y=(1, 0) 形状约定对齐；
            # RemoveYTransform 可在推理时将其清零以进行无条件生成。
            y = torch.tensor(
                all_embeddings[global_idx], dtype=torch.float
            ).unsqueeze(0)   # (1, 1024)

            data = Data(
                x            = x,
                edge_index   = edge_index,
                edge_attr    = edge_attr,
                y            = y,
                smiles       = smi,
                precursor_mz = torch.tensor(
                    [float(all_precursor[global_idx])], dtype=torch.float
                ),
                idx          = int(global_idx),
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        print(
            f"[{self.stage}] Saved molecules: {len(data_list)} | "
            f"Invalid SMILES: {n_invalid} | "
            f"Filtered (atoms/bonds out of vocabulary): {n_filtered}"
        )
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])


# ══════════════════════════════════════════════════════════════════
#  MetaboliteDataModule
# ══════════════════════════════════════════════════════════════════

class MetaboliteDataModule(MolecularDataModule):
    """
    管理三折数据集，与 QM9DataModule 结构完全对齐。

    cfg 中需要包含：
        cfg.dataset.datadir   : 缓存目录（相对项目根目录）
        cfg.dataset.hdf5_path : HDF5 文件路径
        cfg.dataset.remove_h  : bool
    """

    def __init__(self, cfg):
        self.datadir   = cfg.dataset.datadir
        self.hdf5_path = cfg.dataset.hdf5_path
        self.remove_h  = cfg.dataset.remove_h

        base_path  = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path  = os.path.join(base_path, self.datadir)

        # guidance 阶段：train/val 保留 y（DreaMS embedding）
        # test 阶段也保留，方便计算条件生成指标
        datasets = {
            'train': MetaboliteDataset(
                stage='train', root=root_path, hdf5_path=self.hdf5_path,
                remove_h=self.remove_h, transform=KeepDreaMSTransform()),
            'val':   MetaboliteDataset(
                stage='val',   root=root_path, hdf5_path=self.hdf5_path,
                remove_h=self.remove_h, transform=KeepDreaMSTransform()),
            'test':  MetaboliteDataset(
                stage='test',  root=root_path, hdf5_path=self.hdf5_path,
                remove_h=self.remove_h, transform=KeepDreaMSTransform()),
        }
        super().__init__(cfg, datasets)


# ══════════════════════════════════════════════════════════════════
#  MetaboliteInfos
# ══════════════════════════════════════════════════════════════════

def _fold_n_nodes_histogram(datamodule, cap: int) -> torch.Tensor:
    """节点数分布（概率），长度 cap；超大图折叠进最后一档。"""
    raw = datamodule.node_counts(max_nodes_possible=max(300, cap))
    if raw.sum() <= 0:
        return torch.ones(cap) / cap
    if raw.numel() > cap:
        out = raw[:cap].clone()
        out[-1] = out[-1] + raw[cap:].sum()
    else:
        out = torch.zeros(cap)
        out[: raw.numel()] = raw
    return out / out.sum()


class MetaboliteInfos(AbstractDatasetInfos):
    """
    数据集元信息。

    默认 cfg.dataset.metabolite_use_data_statistics=true 时，n_nodes / node_types /
    edge_types / valency_distribution 从训练 DataLoader 统计，与 marginal 扩散一致。

    若需「打印数值再手工贴回代码」的旧流程：设 metabolite_use_data_statistics=false，
    并设 general.recompute_statistics=true 跑一次（仍会 assert 退出）。
    """

    def __init__(self, datamodule, cfg, recompute_statistics: bool = False):
        self.remove_h = cfg.dataset.remove_h
        self.name     = 'metabolite'
        self.need_to_strip = False   # DiGress 内部标志
        use_data_stats = getattr(cfg.dataset, "metabolite_use_data_statistics", True) and not recompute_statistics
        root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
        suffix = 'no_h' if self.remove_h else 'h'
        stats_cache_path = os.path.join(root_dir, cfg.dataset.datadir, f'metabolite_stats_{suffix}.pt')
        cached_stats = None
        save_stats_cache = False

        if use_data_stats and os.path.exists(stats_cache_path):
            try:
                payload = torch.load(stats_cache_path, weights_only=False)
                if payload.get('max_n_nodes') == int(getattr(cfg.dataset, "max_n_nodes", 100 if self.remove_h else 200)):
                    cached_stats = payload
                    print(f"[MetaboliteInfos] Loaded statistics cache: {stats_cache_path}")
            except Exception as exc:
                print(f"[MetaboliteInfos] Failed to read cache, recomputing statistics: {exc}")
        elif use_data_stats:
            print(
                "[MetaboliteInfos] Statistics cache not found. "
                "Computing dataset statistics from dataloaders (one-time, may take a long time)..."
            )

        if self.remove_h:
            self.atom_encoder   = {sym: i for i, sym in enumerate(ATOM_TYPES_NO_H)}
            self.atom_decoder   = ATOM_TYPES_NO_H          # ['C','N','O','S','P','F','Cl','Br','I']
            self.num_atom_types = len(ATOM_TYPES_NO_H)     # 9
            self.valencies      = VALENCIES_NO_H            # [4,3,2,6,5,1,1,1,1]
            self.atom_weights   = ATOM_WEIGHTS_NO_H
            self.max_n_nodes    = int(getattr(cfg.dataset, "max_n_nodes", 100))
            self.max_weight     = 1500   # 分子量上限（Da）

            cap = self.max_n_nodes + 1
            if use_data_stats:
                if cached_stats is not None:
                    self.n_nodes = cached_stats['n_nodes']
                    self.node_types = cached_stats['node_types']
                    self.edge_types = cached_stats['edge_types']
                else:
                    self.n_nodes = _fold_n_nodes_histogram(datamodule, cap)
                    self.node_types = datamodule.node_types()
                    self.edge_types = datamodule.edge_counts()
                    save_stats_cache = True
            else:
                self.n_nodes = torch.zeros(cap)
                self.node_types = torch.tensor([
                    0.650, 0.090, 0.180, 0.020, 0.010, 0.010, 0.020, 0.010, 0.010,
                ])
                self.edge_types = torch.tensor([0.0, 0.62, 0.16, 0.02, 0.20])

        else:
            self.atom_encoder   = {sym: i for i, sym in enumerate(ATOM_TYPES_H)}
            self.atom_decoder   = ATOM_TYPES_H
            self.num_atom_types = len(ATOM_TYPES_H)        # 10
            self.valencies      = VALENCIES_H               # [1,4,3,2,6,5,1,1,1,1]
            self.atom_weights   = ATOM_WEIGHTS_H
            self.max_n_nodes    = int(getattr(cfg.dataset, "max_n_nodes", 200))
            self.max_weight     = 1500

            cap = self.max_n_nodes + 1
            if use_data_stats:
                if cached_stats is not None:
                    self.n_nodes = cached_stats['n_nodes']
                    self.node_types = cached_stats['node_types']
                    self.edge_types = cached_stats['edge_types']
                else:
                    self.n_nodes = _fold_n_nodes_histogram(datamodule, cap)
                    self.node_types = datamodule.node_types()
                    self.edge_types = datamodule.edge_counts()
                    save_stats_cache = True
            else:
                self.n_nodes = torch.zeros(cap)
                self.node_types = torch.ones(self.num_atom_types) / self.num_atom_types
                self.edge_types = torch.tensor([0.0, 0.75, 0.10, 0.01, 0.14])

        # 关闭 use_data_stats 且 n_nodes 仍为全零时，避免 Categorical NaN
        if (not use_data_stats) and self.n_nodes.sum() == 0 and not recompute_statistics:
            self.n_nodes = _fold_n_nodes_histogram(datamodule, self.max_n_nodes + 1)

        # 初始化父类（计算 input_dims 等）
        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

        if not recompute_statistics:
            if use_data_stats:
                if cached_stats is not None and 'valency_distribution' in cached_stats:
                    self.valency_distribution = cached_stats['valency_distribution']
                else:
                    self.valency_distribution = datamodule.valency_count(self.max_n_nodes)
                    save_stats_cache = True
            else:
                self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
                # Keep metrics initialization valid when using placeholder statistics.
                # HistogramsMAE expects a normalized target histogram with sum ~= 1.
                self.valency_distribution[1:6] = torch.tensor([0.20, 0.35, 0.25, 0.15, 0.05])

        if use_data_stats and (not recompute_statistics) and save_stats_cache:
            os.makedirs(os.path.dirname(stats_cache_path), exist_ok=True)
            torch.save({
                'max_n_nodes': self.max_n_nodes,
                'n_nodes': self.n_nodes,
                'node_types': self.node_types,
                'edge_types': self.edge_types,
                'valency_distribution': self.valency_distribution,
            }, stats_cache_path)
            print(f"[MetaboliteInfos] Statistics cache saved: {stats_cache_path}")

        if (not recompute_statistics) and self.valency_distribution.sum() <= 0:
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[1] = 1.0
            print("[MetaboliteInfos] Warning: valency distribution was empty; using a safe fallback.")

        # ── 统计重算（仅打印 + 写 txt；assert 退出）──────────────────
        if recompute_statistics:
            np.set_printoptions(suppress=True, precision=5)

            self.n_nodes = datamodule.node_counts()
            print("n_nodes =", self.n_nodes)
            np.savetxt('n_counts.txt', self.n_nodes.numpy())

            self.node_types = datamodule.node_types()
            print("node_types =", self.node_types)
            np.savetxt('atom_types.txt', self.node_types.numpy())

            self.edge_types = datamodule.edge_counts()
            print("edge_types =", self.edge_types)
            np.savetxt('edge_types.txt', self.edge_types.numpy())

            valencies = datamodule.valency_count(self.max_n_nodes)
            print("valency_distribution =", valencies)
            np.savetxt('valencies.txt', valencies.numpy())
            self.valency_distribution = valencies

            # 用真实统计重新初始化父类
            super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

            # QM9 惯例：统计完成后强制停止，提示用户将数值填入上方占位符
            assert False, (
                "Statistics recomputation finished. Please copy the printed tensors "
                "into the MetaboliteInfos placeholders, then set "
                "recompute_statistics=False and run again."
            )


# ══════════════════════════════════════════════════════════════════
#  get_train_smiles（与 QM9 对应）
# ══════════════════════════════════════════════════════════════════

def get_train_smiles(cfg, train_dataloader, dataset_infos, evaluate_dataset=False):
    """
    提取训练集 SMILES 列表（用于新颖性/唯一性指标计算）。
    结果缓存为 .npy 文件，避免重复计算。
    """
    if evaluate_dataset:
        assert dataset_infos is not None

    datadir      = cfg.dataset.datadir
    remove_h     = cfg.dataset.remove_h
    atom_decoder = dataset_infos.atom_decoder

    root_dir    = pathlib.Path(os.path.realpath(__file__)).parents[2]
    suffix      = 'no_h' if remove_h else 'h'
    smiles_path = os.path.join(root_dir, datadir, f'train_smiles_{suffix}.npy')

    if os.path.exists(smiles_path):
        print("Cached train SMILES found, loading...")
        train_smiles = np.load(smiles_path, allow_pickle=True)
    else:
        print("Extracting train SMILES from dataloader...")
        train_smiles = _extract_smiles_from_dataloader(
            atom_decoder, train_dataloader, remove_h
        )
        np.save(smiles_path, np.array(train_smiles))

    if evaluate_dataset:
        all_molecules = []
        for data in train_dataloader:
            dense_data, node_mask = utils.to_dense(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E
            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                all_molecules.append([X[k, :n].cpu(), E[k, :n, :n].cpu()])

        print(f"Evaluating dataset — {len(all_molecules)} molecules")
        metrics = compute_molecular_metrics(
            molecule_list=all_molecules,
            train_smiles=train_smiles,
            dataset_info=dataset_infos,
        )
        print(metrics[0])

    return train_smiles


def _extract_smiles_from_dataloader(atom_decoder, dataloader, remove_h: bool):
    """从 PyG Dense 批次重建 SMILES（与 compute_qm9_smiles 对应）"""
    mols_smiles  = []
    n_invalid = n_disconnected = 0

    for i, data in enumerate(dataloader):
        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E

        for k in range(X.size(0)):
            n   = int(torch.sum((X != -1)[k, :]))
            mol = build_molecule_with_partial_charges(
                X[k, :n].cpu(), E[k, :n, :n].cpu(), atom_decoder
            )
            smi = mol2smiles(mol)
            if smi is not None:
                mols_smiles.append(smi)
                frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                if len(frags) > 1:
                    n_disconnected += 1
            else:
                n_invalid += 1

        if i % 200 == 0:
            print(f"\tExtracting SMILES: {i}/{len(dataloader)}")

    print(f"Invalid: {n_invalid} | Disconnected: {n_disconnected}")
    return mols_smiles
