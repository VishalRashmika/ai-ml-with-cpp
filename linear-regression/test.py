# import matplotlib.pyplot as plt
# import pandas as pd

# df = pd.read_csv("homeprices.csv")

# m = 0.974313
# b = 300.181

# # m = 45.2667
# # b = 0.129667

# plt.scatter(df.area,df.price, color="blue")
# plt.plot(list(range(250,500)), [m * x + b for x in range(250,500)], color="red")
# plt.show()
import pandas as pd

points = pd.read_csv('dataset.csv')

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = float(len(points))
    for i in range(len(points)):
        x = points.iloc[i, 0]
        y = points.iloc[i, 1]
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
    m = m_now - L * m_gradient
    b = b_now - L * b_gradient
    return [m, b]

m = 0
b = 0
L = 0.0001
epochs = 100000

for i in range(epochs):
    m, b = gradient_descent(m, b, points, L)

print(m, b)