import numpy as np

# Функция активации (сигмоида) и её производная
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# Данные: XOR
# Вход: 4 примера по 2 признака
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Ожидаемый выход (XOR)
Y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(42)

# Веса: слой 1 (2 входа → 4 нейрона), слой 2 (4 → 1 выход)
W1 = np.random.randn(2, 4)
W2 = np.random.randn(4, 1)

learning_rate = 0.5
epochs = 10000

# Обучение
for epoch in range(epochs):
    # Прямой проход (forward pass)
    layer1 = sigmoid(X @ W1)       # (4, 4)
    output = sigmoid(layer1 @ W2)  # (4, 1)

    # Ошибка
    error = Y - output

    # Обратный проход (backpropagation)
    d_output = error * sigmoid_deriv(output)
    d_layer1 = (d_output @ W2.T) * sigmoid_deriv(layer1)

    # Обновление весов
    W2 += layer1.T @ d_output * learning_rate
    W1 += X.T @ d_layer1 * learning_rate

    if epoch % 1000 == 0:
        loss = np.mean(error ** 2)
        print(f"Эпоха {epoch:5d} | Ошибка: {loss:.4f}")

# Результат
print("\nРезультат после обучения:")
for i, (x, y) in enumerate(zip(X, Y)):
    pred = output[i][0]
    print(f"  Вход: {x} | Ожидалось: {y[0]} | Получилось: {pred:.4f} ({'✓' if round(pred) == y[0] else '✗'})")
