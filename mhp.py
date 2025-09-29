import numpy as np

class MHP:
    def __init__(self, lambda_0, alpha, beta, T):
        """
        初始化 MHP 模型
        :param lambda_0: 背景强度
        :param alpha: 激励参数
        :param beta: 衰减参数
        :param T: 模拟的总时间
        """
        self.lambda_0 = lambda_0  # 背景强度
        self.alpha = alpha  # 激励参数
        self.beta = beta  # 衰减参数
        self.T = T  # 模拟时间
        self.times = []  # 事件发生的时间列表

    def update_intensity(self, t):
        """
        根据历史事件更新强度
        :param t: 当前时间
        :return: 当前的强度值
        """
        intensity = self.lambda_0
        for prev_t in self.times:
            intensity += self.alpha * np.exp(-self.beta * (t - prev_t))
        return intensity

    def simulate(self):
        """
        使用 Ogata 改进的薄化算法模拟 Hawkes 过程
        :return: 事件发生的时间列表
        """
        t = 0
        while t < self.T:
            # 生成下一个潜在事件的时间间隔
            dt = np.random.exponential(1 / self.update_intensity(t))
            t += dt

            # 判断是否接受这个事件
            u = np.random.uniform(0, 1)
            if u <= self.update_intensity(t) / self.lambda_0:
                self.times.append(t)
        return self.times

    def plot_events(self):
        """
        绘制事件时间的直方图
        """
        import matplotlib.pyplot as plt
        import matplotlib
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        plt.figure(figsize=(10, 6))
        plt.hist(self.times, bins=50, alpha=0.7, color='g', edgecolor='black')
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('事件计数', fontsize=12)
        plt.title('Hawkes过程事件时间分布', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
