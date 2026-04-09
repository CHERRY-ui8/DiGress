# DiGress 配置说明

本目录为 **Hydra** 配置根。入口文件是 **`config.yaml`**，其中的 `defaults` 列表决定会合并哪些子配置。

---

## 一、当前默认：代谢物（metabolite）

为简化「代谢物条件生成」的日常使用，**默认已改为代谢物数据集 + 较小离散模型 + 对应训练超参**，无需再写 `experiment` 或 `+experiment`。

在项目根目录执行：

```bash
python src/main.py
```

等价于使用：

- `general: general_default` — 运行名、wandb、采样数量等  
- `model: discrete` — 代谢物友好的离散图 Transformer 宽度/层数  
- `train: train_default` — batch、学习率、EMA 等  
- `dataset: metabolite` — 数据路径、`max_n_nodes`、统计开关等  

常用**仅改一两项**时，在命令行覆盖即可，例如：

```bash
python src/main.py train.batch_size=2 dataset.hdf5_path=/path/to/your.hdf5
```

Hydra 会把完整配置写入每次运行目录下的 `outputs/.../.hydra/config.yaml`，便于核对。

---

## 二、原来的用法（上游 DiGress 习惯）

典型写法是：

- 根配置里默认 **`dataset: qm9`**（或其它数据集），  
- 再通过 **`+experiment=xxx`** 追加一组实验配置（见 `experiment/` 目录下的各 yaml）。

`experiment/*.yaml` 里常带 `# @package _global_`，表示一次同时覆盖 **general / train / model / dataset** 等多块，相当于「一条实验配方」。

**与 `model/discrete.yaml` 的关系**：并不矛盾。`model/discrete` 只负责 **`cfg.model`**；`experiment` 配方里也可以写 `model:` 下的字段，在 Hydra 合并顺序里**后加载的会覆盖先加载的**。以前容易混淆，多半是因为「defaults 里已有 `model: discrete`，又叠加了 experiment」，需要看清合并顺序和是否写进全局。

---

## 三、如何切回 QM9 风格（小图、大 batch、原默认模型）

若需要按原仓库思路跑 **QM9**，可显式指定数据集与配套的三份配置（避免与代谢物默认混在一起）：

```bash
python src/main.py dataset=qm9 general=general_qm9 train=train_qm9 model=discrete_qm9
```

---

## 四、`experiment/` 目录

- **`experiment/`** 下仍保留 **SBM、planar、QM9 变体**等**历史/其它任务**的配方（例如 `sbm.yaml`）供参考或单独启用。  
- **当前默认训练代谢物时，不必依赖这些文件**；若要在新默认流程里复用某条 experiment，需在 **`config.yaml` 的 `defaults` 里显式加入**对应配置组，并理解 Hydra 的覆盖顺序（此处不再展开）。

---

## 五、小结

| 场景 | 做法 |
|------|------|
| 日常训代谢物 | `python src/main.py`（必要时加命令行覆盖） |
| 训 QM9 | 使用上面「切回 QM9」一条命令 |
| 改模型结构 | 编辑 `model/discrete.yaml` 或新增 `model/xxx.yaml` 并在命令行 `model=xxx` |
| 核对实际生效配置 | 查看本次运行目录下 `.hydra/config.yaml` |
