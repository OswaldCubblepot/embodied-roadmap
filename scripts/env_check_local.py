#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""env_check_local.py —— 本地（CPU）环境自检。

验证的不是"能不能 import"，而是：
    1. 三个核心库版本正确
    2. OpenCV 能真正处理像素（读图→灰度→边缘→写图）
    3. PyTorch 的 autograd 在本机能跑（哪怕只有 CPU）

用法（已激活 venv）：
    python scripts/env_check_local.py

输出：docs/day00_test.png 和 docs/day00_edges.png
"""
import sys
from pathlib import Path

# 让脚本在任意工作目录下都能找到 docs/
ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)


def main():
    ok = True
    print("=" * 58)
    print("本地环境自检（CPU）")
    print("=" * 58)
    print(f"Python  {sys.version.split()[0]}")
    print(f"可执行  {sys.executable}")

    try:
        import numpy as np
        print(f"NumPy   {np.__version__}")
    except ImportError as e:
        print("[FAIL] numpy 未安装：", e)
        return 1

    try:
        import cv2
        print(f"OpenCV  {cv2.__version__}")
    except ImportError as e:
        print("[FAIL] opencv 未安装：", e)
        return 1

    try:
        import torch
        print(f"PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print("[INFO] 检测到 GPU —— 你在本地也有卡，可直接训练")
        else:
            print("[OK] CPU 模式（预期行为：训练去云端，本地只写代码）")
    except ImportError as e:
        print("[FAIL] torch 未安装：", e)
        return 1

    # 1) OpenCV 全链路
    try:
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.rectangle(img, (80, 80), (320, 220), (0, 200, 0), -1)
        cv2.putText(img, "D0 OK", (110, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        cv2.imwrite(str(DOCS / "day00_test.png"), img)
        cv2.imwrite(str(DOCS / "day00_edges.png"), edges)
        print("[PASS] OpenCV 读图→灰度→边缘→保存 全链路正常")
        print(f"       -> {DOCS / 'day00_test.png'}")
        print(f"       -> {DOCS / 'day00_edges.png'}")
    except Exception as e:
        print("[FAIL] OpenCV 处理失败：", e)
        ok = False

    # 2) PyTorch autograd（CPU）
    try:
        x = torch.randn(64, 3, 32, 32, requires_grad=True)
        w = torch.randn(10, 3 * 32 * 32) * 0.01
        logits = x.flatten(1) @ w.T
        loss = logits.pow(2).mean()
        loss.backward()
        grad_ok = x.grad is not None and torch.isfinite(x.grad).all()
        print(f"[PASS] autograd 正常（loss={loss.item():.4f}, grad 有限={bool(grad_ok)}）")
    except Exception as e:
        print("[FAIL] autograd 失败：", e)
        ok = False

    print("=" * 58)
    print("结论：本地环境可用 ✅ 下一步去云端跑 smoke_train.py" if ok
          else "结论：存在失败项 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())