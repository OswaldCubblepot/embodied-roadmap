#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_cifar.py —— D1：手写 CIFAR-10 训练循环（骨架）

你的任务：把 3 个 TODO 补完，让 test acc >= 85%。

硬规矩（这条不许绕过）：
    · 不许用 pytorch-lightning / 任何 Trainer / model.fit()
    · 不许用 torchvision.models 的预训练权重
    · 必须自己写 for epoch、自己算 loss、自己 optimizer.step()

跑法：
    # 本地冒烟（CPU 也能跑，1 分钟）
    python scripts/train_cifar.py --epochs 1 --limit 2000

    # 云端正式
    python scripts/train_cifar.py --epochs 30

第一次运行会自动下载 CIFAR-10（约 170MB）。
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# --------------------------------------------------------------------------
# 数据管道（已写好，看懂即可 —— 任务 1 就是读懂这段）
# --------------------------------------------------------------------------
# CIFAR-10 的通道均值和标准差。注意：不是 ImageNet 的 (0.485, 0.456, 0.406)！
# 用错的话 accuracy 会莫名其妙低 5-10 个点。
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def get_loaders(data_root="./data", bs=128, workers=4, limit=0):
    """返回 (train_loader, test_loader)。

    训练集做数据增强，测试集只做 ToTensor + Normalize。
    这是标准做法：增强只作用于训练，测试必须"原样"评估。
    """
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # 随机裁剪：让模型对位置不敏感
        transforms.RandomHorizontalFlip(),      # 随机翻转：让模型对左右不敏感
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_set = datasets.CIFAR10(data_root, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)

    # --limit 用于冒烟测试：只取前 N 张，本地 CPU 也能快速跑完一个 epoch
    if limit > 0:
        train_set = Subset(train_set, range(min(limit, len(train_set))))
        test_set = Subset(test_set, range(min(limit // 5, len(test_set))))

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=workers, pin_memory=torch.cuda.is_available(),
                              drop_last=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=workers, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader


# --------------------------------------------------------------------------
# 模型（已写好，P2 会详细讲；今天你只需要知道输入输出形状）
# --------------------------------------------------------------------------
class BasicBlock(nn.Module):
    """残差块：out = relu(conv2(relu(conv1(x))) + shortcut(x))"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # shortcut：维度不匹配时用 1x1 卷积把通道数和分辨率对齐
        self.short = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.short(x)      # <-- 这行就是"残差"
        return self.relu(out)


class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = BasicBlock(32, 64, stride=1)
        self.layer2 = BasicBlock(64, 128, stride=2)
        self.layer3 = BasicBlock(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# --------------------------------------------------------------------------
# TODO ①：训练一轮
# --------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, log_every=50):
    """跑完一整轮训练，返回 (平均loss, 准确率)。

    提示（顺序错了就白跑）：
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)   # ① 清梯度
            out = model(x)                          # ② 前向
            loss = criterion(out, y)                # ③ 损失
            loss.backward()                         # ④ 反向
            optimizer.step()                        # ⑤ 更新

    要统计的量：
        total_loss  += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
    最后除以样本总数得平均值。

    加分项（可选）：用 scaler 做混合精度
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(x); loss = criterion(out, y)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    """
    raise NotImplementedError("TODO ①：写 train_one_epoch")


# --------------------------------------------------------------------------
# TODO ②：评估
# --------------------------------------------------------------------------
def evaluate(model, loader, criterion, device):
    """在测试集上评估，返回 (平均loss, 准确率)。

    三个必须：
        model.eval()            —— 关掉 Dropout、让 BN 用滑动统计量
        torch.no_grad()         —— 不建计算图，省显存也更快
        不要调用 optimizer.step()
    """
    raise NotImplementedError("TODO ②：写 evaluate")


# --------------------------------------------------------------------------
# 画图（已写好）
# --------------------------------------------------------------------------
def plot_curves(history, out_path):
    import matplotlib
    matplotlib.use("Agg")          # 服务器上无显示器，必须用非交互后端
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep, history["train_loss"], label="train")
    axes[0].plot(ep, history["test_loss"], label="test")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend(); axes[0].set_title("Loss")

    axes[1].plot(ep, history["train_acc"], label="train")
    axes[1].plot(ep, history["test_acc"], label="test")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy (%)")
    axes[1].legend(); axes[1].set_title("Accuracy")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"[saved] {out_path}")


# --------------------------------------------------------------------------
# TODO ③：训练主循环
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=5e-4, help="weight decay")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="只用前 N 张图（冒烟测试）")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out", default="docs")
    p.add_argument("--amp", action="store_true", help="开启混合精度")
    p.add_argument("--log-every", type=int, default=50, help="每 N 个 step 打印一次")
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)   # 否则 torch.save 会报"父目录不存在"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    train_loader, test_loader = get_loaders(args.data_root, args.bs, args.workers, args.limit)
    print(f"train batches = {len(train_loader)}, test batches = {len(test_loader)}")

    model = SmallResNet().to(device)
    criterion = nn.CrossEntropyLoss()

    # ---- TODO ③-a：创建优化器和学习率调度器 ----
    # 推荐：AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    #       CosineAnnealingLR(optimizer, T_max=args.epochs)
    # 想对比的话，可以换成 SGD(momentum=0.9)，记录两者差异（任务 4 第 3 题要用）
    raise NotImplementedError("TODO ③-a：写 optimizer 和 scheduler")

    # ---- TODO ③-b：训练循环 ----
    # 每一轮：
    #   1) tr_loss, tr_acc = train_one_epoch(...)   # 记得把 args.log_every 传进去
    #   2) te_loss, te_acc = evaluate(...)
    #   3) scheduler.step()
    #   4) 打印日志（带耗时、当前 lr）
    #   5) 若 te_acc 是历史最好 -> 保存 best.pth
    #   6) 把四个指标 append 进 history
    raise NotImplementedError("TODO ③-b：写训练主循环")

    # ---- 收尾（已写好，等 history 填好后自动生效）----
    # plot_curves(history, f"{args.out}/curves.png")
    # json.dump(history, open(f"{args.out}/history.json", "w"))
    # print(f"best test acc = {best_acc:.2f}%")


if __name__ == "__main__":
    main()