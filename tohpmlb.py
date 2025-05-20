def update_w_and_b(X, Y, w, b, alpha):
    """
    Updates the weights (w) and bias (b) of a linear model using gradient descent.

        Args:
            X: The input features.
            Y: The target values.
            w: The current weight value.
            b: The current bias value.
            alpha: The learning rate.

        Returns:
            tuple: A tuple containing the updated weight (w) and bias (b).
    """
    dl_dw = 0.0
    dl_db = 0.0

    N = len(X)

    for i in range(N):
        dl_dw += -2 * X[i] * (Y[i] - (w * X[i] + b))
        dl_db += -2 * (Y[i] - (w * X[i] + b))

    w = w - (1 / float(N)) * dl_dw * alpha
    b = b - (1 / float(N)) * dl_db * alpha

    return w, b


spendings = [0, 2, 3, 5, 8, 11]
sales = [0, 0, 1, 1, 2, 3]
alpha = 0.001
epochs = 4444
(w, b) = (0, 0)

w, b = update_w_and_b(
    spendings,
    sales,
    w,
    b,
    alpha=0.001,
)
print(w, b)


def avg_loss(X, Y, w, b):
    """
    Calculates the average squared loss.

        Args:
            X: Input data.
            Y: Target values.
            w: Weight parameter.
            b: Bias parameter.

        Returns:
            float: The average squared loss value.
    """
    N = len(X)
    total_error = 0.0
    for i in range(N):
        total_error += (Y[i] - (w * X[i] + b)) ** 2
    return total_error / float(N)


def train(X, Y, w, b, alpha, epochs):
    """
    Trains the model for a specified number of epochs.

        Args:
            X: The input features.
            Y: The target values.
            w: The initial weight vector.
            b: The initial bias term.
            alpha: The learning rate.
            epochs: The number of training iterations.

        Returns:
            tuple: A tuple containing the updated weight vector and bias term after training.
    """
    for e in range(epochs):
        w, b = update_w_and_b(X, Y, w, b, alpha)
        # log the progress
        if e % 400 == 0:
            print("epoch:", e, "loss: ", avg_loss(X, Y, w, b))
    return w, b


train(spendings, sales, w, b, alpha, epochs)
