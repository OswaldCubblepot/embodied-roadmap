# Embodied AI Roadmap

具身智能 / 机器人视觉学习与实验记录。目标：2027 年拿到 VLA / 三维视觉方向实习。

## 我在做什么
- 2026.11–2027.06：从零补 PyTorch / 计算机视觉 / 模仿学习 → VLA 复现 → 真机部署
- 主线项目：① 位姿估计工具 ② 仿真抓取策略 ③ VLA 复现与消融 ④ 闭环部署

## 目录
- `weekly/`  每周复盘（做了啥、卡在哪、下周计划）
- `scripts/` 可复用脚本
- `notes/`   论文笔记（一篇一页）
- `docs/`    图表、截图、报告

## 环境

两套环境，分工明确：**本地写代码，云端跑训练**。

| | 本地（Windows / CPU） | 云端（Linux / RTX 4090） |
|---|---|---|
| 用途 | 写代码、调 OpenCV、跑小实验 | 训练、微调、批量数据处理 |
| PyTorch | 2.15.0+cpu | 2.14.1+cu130 |
| 依赖 | `requirements-local.txt` | `requirements-cloud.txt` |
| 自检 | `python scripts/env_check_local.py` | `python scripts/smoke_train.py` |

详细实测数据见 [`docs/env-2026-11-04.md`](docs/env-2026-11-04.md)。

### 本地

```powershell
# 推荐：直接指定解释器，不用管 PowerShell 执行策略
.\.venv\Scripts\python.exe scripts\env_check_local.py

# 或者激活后用 python（首次可能需解除脚本限制）
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 云端

```bash
# 非交互式 SSH 不会加载 .bashrc，必须手动激活
source /usr/local/miniconda3/bin/activate py310
python scripts/smoke_train.py
```

## 进度
| 阶段         | 时间            | 状态     |
| ------------ | --------------- | -------- |
| P0 地基      | 2026.11         | 🟡 进行中 |
| P1 视觉几何  | 2026.12         | ⬜        |
| P2 模仿学习  | 2026.12–2027.02 | ⬜        |
| P3 VLA 主线  | 2027.02–04      | ⬜        |
| P4 真机/部署 | 2027.04–05      | ⬜        |