from tick.hawkes import HawkesProcess
from tick.hawkes import SimuHawkesExpKernels

# 尝试加载 Bund 数据集
try:
    dataset = SimuHawkesExpKernels.from_dataset('bund')
    print("Bund 数据集加载成功！")
except Exception as e:
    print(f"加载 Bund 数据集时出错：{e}")
