"""D0 环境自检：读图 → 灰度 → 边缘 → 保存。跑通即环境 OK。"""
import sys
import cv2
import numpy as np
import torch

print(f"Python  {sys.version.split()[0]}")
print(f"NumPy   {np.__version__}")
print(f"OpenCV  {cv2.__version__}")
print(f"PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}")

img = np.zeros((300, 400, 3), dtype=np.uint8)
cv2.rectangle(img, (80, 80), (320, 220), (0, 200, 0), -1)
cv2.putText(img, "D0 OK", (110, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

cv2.imwrite("docs/day00_test.png", img)
cv2.imwrite("docs/day00_edges.png", edges)
print("saved -> docs/day00_test.png, docs/day00_edges.png")