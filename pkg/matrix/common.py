import numpy as np
import matplotlib.pyplot as plt

def visualize_matrix_iteration(M, n_points=8, T=10, xlim=(-2,2), ylim=(-2,2)):
    """
    可视化二维矩阵 M 对多个向量的迭代变换
    M: 2x2 numpy array
    n_points: 每维起始点个数（总点数 n_points*n_points）
    T: 迭代步数
    """
    # 生成初始网格点
    xs = np.linspace(xlim[0], xlim[1], n_points)
    ys = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(xs, ys)
    start_points = np.stack([X.ravel(), Y.ravel()], axis=1)

    plt.figure(figsize=(8,8))
    colors = plt.cm.viridis(np.linspace(0, 1, T+1))  # 时间颜色映射

    for start in start_points:
        v = start.copy()
        points = [v.copy()]
        for t in range(T):
            v = M @ v
            points.append(v.copy())
        points = np.array(points)
        # 绘制轨迹（折线），颜色按时间渐变
        for i in range(T):
            plt.plot(points[i:i+2, 0], points[i:i+2, 1],
                     color=colors[i], alpha=0.7, linewidth=1.5)
        # 标记起点（绿色圆点）和终点（红色星号）
        plt.plot(points[0,0], points[0,1], 'go', markersize=5)
        plt.plot(points[-1,0], points[-1,1], 'r*', markersize=8)

    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(f"Matrix Iteration: {T} steps\nEigenvalues: {np.linalg.eigvals(M)}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(alpha=0.3)
    plt.show()

# 收缩矩阵（收敛到0）
# M1 = np.array([[0.7, 0.1], [0.1, 0.6]])
# visualize_matrix_iteration(M1, n_points=6, T=15)

# 特征值1和0.5（收敛到非零向量）
M2 = np.array([[1, 0.2], [0, 0.5]])
visualize_matrix_iteration(M2, n_points=6, T=20)

# # 旋转矩阵（不收敛）
# theta = np.radians(30)
# M3 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
# visualize_matrix_iteration(M3, n_points=6, T=20)
#
# # 发散矩阵（限制显示范围，只画几步）
# M4 = np.array([[1.2, 0], [0, 0.8]])
# visualize_matrix_iteration(M4, n_points=5, T=10, xlim=(-3,3), ylim=(-3,3))