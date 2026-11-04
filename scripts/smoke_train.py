#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""smoke_train.py —— 真正的训练冒烟测试。

验证的不是"能不能 import"，而是：
    1. 前向 + 反向 + 优化器 三步都通
    2. 显存峰值在 4090 的合理范围
    3. 训练速度能支撑你后面的实验节奏
    4. matmul 峰值算力（预热后测量，避免冷卡误判）

用法：python smoke_train.py
"""
import time

import torch
import torch.nn as nn


def bench_matmul(n=4096, iters=100):
    print("=" * 62)
    print("1) matmul 峰值算力（预热后）")
    x = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(10):            # 预热：让 GPU 从 P8 升到 P0 并加载 cublas kernel
        _ = x @ x
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = x @ x
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    tflops = 2 * n ** 3 / dt / 1e12
    print(f"   bf16 {n}x{n}: {dt * 1000:.2f} ms/iter  ->  {tflops:.1f} TFLOPS")

    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    for _ in range(10):
        _ = a @ a
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = a @ a
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    print(f"   fp16 {n}x{n}: {dt * 1000:.2f} ms/iter  ->  {2 * n ** 3 / dt / 1e12:.1f} TFLOPS")
    return tflops


class TinyNet(nn.Module):
    """一个够小的 CNN，纯粹用来验证训练链路。"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def smoke_train(steps=80, bs=64):
    print("=" * 62)
    print(f"2) 训练冒烟测试（TinyNet, bs={bs}, {steps} steps）")
    model = TinyNet().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # 固定一批随机数据，故意让网络去"过拟合"它 —— loss 必须明显下降
    x = torch.randn(bs, 3, 64, 64, device="cuda")
    y = torch.randint(0, 10, (bs,), device="cuda")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    losses = []
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(x)
            loss = loss_fn(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
    torch.cuda.synchronize()
    dt = time.time() - t0

    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"   {steps} steps 用时 {dt:.2f}s  ->  {steps / dt:.1f} step/s")
    print(f"   loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"   显存峰值: {peak:.2f} GiB")

    ok = losses[-1] < losses[0] * 0.8
    print(("   [PASS] 反向传播与优化器正常，loss 明显下降" if ok
           else "   [WARN] loss 未明显下降，检查 lr / 数据"))
    return ok, peak


def main():
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"{torch.cuda.get_device_name(0)}")
    tflops = bench_matmul()
    ok, peak = smoke_train()
    print("=" * 62)
    print("训练链路可用 ✅" if ok else "训练链路存疑 ⚠️")
    print(f"参考：bf16 峰值 {tflops:.1f} TFLOPS｜训练显存 {peak:.2f} GiB")
    print("下次开机先跑这个脚本，30 秒确认环境没坏。")


if __name__ == "__main__":
    main()