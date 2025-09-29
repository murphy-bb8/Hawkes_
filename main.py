from mhp import MHP

# 定义模型参数
lambda_0 = 0.5  # 背景强度
alpha = 1.0  # 激励参数
beta = 1.0  # 衰减参数
T = 100  # 模拟的总时间

# 创建 MHP 对象并模拟
mhp_model = MHP(lambda_0, alpha, beta, T)
times = mhp_model.simulate()

# 绘制结果
mhp_model.plot_events()
