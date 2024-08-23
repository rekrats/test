import torch
print(torch.__version__)  # 检查 PyTorch 版本
print(torch.cuda.is_available())  # 检查 GPU 是否可用

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # 打印 GPU 名称
    print("Number of GPUs:", torch.cuda.device_count())  # 打印 GPU 数量
    print("Current GPU index:", torch.cuda.current_device())  # 当前使用的 GPU 编号
    print("CUDA version:", torch.version.cuda)  # CUDA 版本
