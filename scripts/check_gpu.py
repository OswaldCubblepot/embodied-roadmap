#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""check_gpu.py —— 云实例开机自检：确认驱动 / CUDA / PyTorch 三者对齐，且 kernel 真能跑。

用法（云端，已激活 conda 环境）：
    python check_gpu.py

判读标准：
    capability (8, 9)      -> RTX 4090 (Ada)，正确
    bf16 matmul OK         -> torch 有该卡的 kernel，可以用
    报 no kernel image     -> torch 版本与显卡架构不匹配，需换 cu12x wheel
"""
import sys
import time

PASS, FAIL = "[PASS]", "[FAIL]"


def main():
    ok = True
    print("=" * 62)
    print("GPU 环境自检")
    print("=" * 62)

    # 1) PyTorch 基本导入
    try:
        import torch
    except Exception as e:
        print(FAIL, "import torch 失败：", e)
        return 1
    print(f"torch        : {torch.__version__}")
    print(f"cuda build   : {torch.version.cuda}")
    print(f"cudnn        : {torch.backends.cudnn.version()}")

    # 2) CUDA 可用性
    if not torch.cuda.is_available():
        print(FAIL, "torch.cuda.is_available() = False（装成 CPU 版或驱动不可用）")
        return 1
    print(PASS, "CUDA 可用")
    print(f"device       : {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"capability   : {cap}")
    if cap != (8, 9):
        print(f"[WARN] 该卡算力为 {cap}，与 4090 的 (8, 9) 不同——如果是别的卡可忽略")

    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    print(f"显存         : {total:.1f} GiB")

    # 3) 真正跑一次 bf16 矩阵乘（比 is_available() 可信）
    try:
        n = 4096
        x = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            y = x @ x
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 20
        tflops = 2 * n ** 3 / dt / 1e12
        print(PASS, f"bf16 matmul OK -> {y.dtype} | {dt * 1000:.1f} ms/iter | {tflops:.1f} TFLOPS")
    except Exception as e:
        print(FAIL, "bf16 matmul 失败：", e)
        print("      常见原因：torch 与该卡架构不匹配（需 cu12x wheel）")
        ok = False

    # 4) fp16 与 fp32 各跑一次，确认真实训练精度可用
    for dt_name, dt in (("fp16", torch.float16), ("fp32", torch.float32)):
        try:
            a = torch.randn(2048, 2048, device="cuda", dtype=dt)
            b = a @ a
            torch.cuda.synchronize()
            print(PASS, f"{dt_name} matmul OK")
        except Exception as e:
            print(FAIL, f"{dt_name} matmul 失败：", e)
            ok = False

    # 5) 显存分配与释放
    try:
        big = torch.empty(int(1.5 * 2 ** 30), dtype=torch.uint8, device="cuda")
        used = torch.cuda.memory_allocated() / 2 ** 30
        del big
        torch.cuda.empty_cache()
        print(PASS, f"显存分配 OK（申请 1.5 GiB，峰值 {used:.2f} GiB）")
    except Exception as e:
        print(FAIL, "显存分配失败：", e)
        ok = False

    # 6) 可选依赖
    print("-" * 62)
    for mod in ("numpy", "cv2", "PIL"):
        try:
            m = __import__(mod)
            print(PASS, f"{mod} {getattr(m, '__version__', '?')}")
        except Exception:
            print("[MISS]", f"{mod} 未安装")

    print("=" * 62)
    print("结论：全部通过 ✅" if ok else "结论：存在问题 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())