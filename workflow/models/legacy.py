import numpy as np


class MHP:
    def __init__(self, lambda_0, alpha, beta, T):
        self.lambda_0 = lambda_0
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.times = []

    def update_intensity(self, t):
        intensity = self.lambda_0
        for prev_t in self.times:
            if t > prev_t:
                intensity += self.alpha * np.exp(-self.beta * (t - prev_t))
        return max(0, intensity)

    def simulate(self):
        t = 0
        self.times = []
        while t < self.T:
            current_intensity = self.update_intensity(t)
            if current_intensity > 0:
                dt = np.random.exponential(1 / current_intensity)
                t += dt
                if t >= self.T:
                    break
                new_intensity = self.update_intensity(t)
                u = np.random.uniform(0, 1)
                if u <= new_intensity / current_intensity:
                    self.times.append(t)
            else:
                t = self.T
        return self.times

    def debug_simulation(self, max_events=1000):
        print("开始调试模拟...")
        print(f"参数: lambda_0={self.lambda_0}, alpha={self.alpha}, beta={self.beta}, T={self.T}")
        t = 0
        self.times = []
        event_count = 0
        while t < self.T and event_count < max_events:
            current_intensity = self.update_intensity(t)
            print(f"时间 t={t:.3f}, 当前强度={current_intensity:.3f}, 事件数={len(self.times)}")
            if current_intensity > 0:
                dt = np.random.exponential(1 / current_intensity)
                t += dt
                if t >= self.T:
                    break
                new_intensity = self.update_intensity(t)
                u = np.random.uniform(0, 1)
                accept_prob = new_intensity / current_intensity if current_intensity > 0 else 0
                print(f"  新时间 t={t:.3f}, 新强度={new_intensity:.3f}, 接受概率={accept_prob:.3f}, u={u:.3f}")
                if u <= accept_prob:
                    self.times.append(t)
                    event_count += 1
                    print(f"  ✓ 接受事件 {event_count} 在时间 {t:.3f}")
                else:
                    print(f"  ✗ 拒绝事件")
            else:
                print("  强度为0，结束模拟")
                t = self.T
        print(f"模拟完成，共生成 {len(self.times)} 个事件")
        return self.times

    def plot_events(self):
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 6))
        plt.hist(self.times, bins=50, alpha=0.7, color='g', edgecolor='black')
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('事件计数', fontsize=12)
        plt.title('Hawkes过程事件时间分布', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


