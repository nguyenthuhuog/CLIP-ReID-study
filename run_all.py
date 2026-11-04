import subprocess
import os
from pathlib import Path

# Đảm bảo thư mục logs tồn tại
Path("logs").mkdir(exist_ok=True)

# Định nghĩa 2 pipeline cần chạy song song
tasks = [
    {
        "name": "CNN",
        "log_path": "logs/cnn.log",
        "cmd": [
            r".venv\Scripts\python.exe", "train_clipreid.py", 
            "--config_file", "configs/person/cnn_clipreid.yml"
        ]
        # Nếu sau train muốn nối tiếp test, bạn có thể dùng dấu && hoặc viết logic nối chuỗi lệnh python ở đây
    },
    {
        "name": "ViT",
        "log_path": "logs/vit.log",
        "cmd": [
            r".venv\Scripts\python.exe", "train_clipreid.py", 
            "--config_file", "configs/person/vit_clipreid.yml"
        ]
    }
]

processes = []

for task in tasks:
    print(f"[INFO] Khoi chay pipeline {task['name']}...")
    log_file = open(task["log_path"], "w", encoding="utf-8")
    
    # Thiết lập môi trường (giữ nguyên CUDA_VISIBLE_DEVICES=0)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Kích hoạt tiến trình ngầm, hướng stdout và stderr vào file log
    p = subprocess.Popen(
        task["cmd"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env
    )
    processes.append((p, log_file))

print("[INFO] Da kich hoat xong tat ca cac tien trinh chay ngam!")
print("[INFO] Ban co the tat cua so nay. Theo doi log tai thu muc logs/")

# (Tùy chọn) Chờ các tiến trình chạy xong nếu muốn giữ terminal mở đến cuối
for p, log_file in processes:
    p.wait()
    log_file.close()