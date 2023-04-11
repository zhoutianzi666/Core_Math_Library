import numpy as np
import matplotlib.pyplot as plt

# 这个是我按照公式写的dft
def my_dft(x):
    N = len(x)
    result = [0] * N
    for k in range(N):
        # 下面就是小n序列
        tmp=np.array(range(N))
        # 下面就是(-j * 2pi * k * n/N)
        tmp = tmp * (-1j) * 2 * np.pi * k * (1 / N)
        tmp = np.exp(tmp)
        result[k] = sum(tmp * x)
    return result

t = np.linspace(0, 1, 1000)
x = np.sin(2*t)
X = np.fft.fft(x)
Y = my_dft(x)

for i in range(len(X)):
    ele1 = X[i]
    ele2 = Y[i]
    if (np.abs(ele1 - ele2) > 0.000001):
        print(ele1)
        print("my_dft 和 np.fft.fft的结果不一样")

# X是一堆复数哦！记住啦！
#print(X)
# 计算序列 x 的频率范围
N = len(x)
# f是横坐标，为啥表示频率范围呢？
f = np.linspace(0, 1, N)

# 绘制序列 x 的频谱图
plt.plot(f, np.abs(X))
plt.xlabel('Frequency (Hz)')
# 下面这个是控制坐标轴范围，但是控制图像大小是哪个选项呢？
#plt.xlim(-1, 1000)
plt.ylabel('Amplitude')
plt.savefig('1.png')
#plt.show()



