import numpy as np
import pandas as pd

def gradient_descent(x, y, lr=0.1, epochs=3000, verbose=100):
    # ----- 0) ensure numpy float arrays -----
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # ----- 1) Min–Max scale x and y -----
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_rng = x_max - x_min
    y_rng = y_max - y_min
    if x_rng == 0:
        raise ValueError("All x values are identical; slope is undefined.")

    x_scaled = (x - x_min) / x_rng
    y_scaled = (y - y_min) / (y_rng if y_rng != 0 else 1.0)  # guard division by 0

    # ----- 2) initialize parameters on scaled space -----
    b = 0.0  # intercept
    m = 0.0  # slope

    # ----- 3) gradient descent on scaled data -----
    for epoch in range(epochs):
        y_hat = b + m * x_scaled                # forward pass
        resid = y_hat - y_scaled                # residuals (ŷ - y)

        # MSE = mean(resid^2); gradients for MSE:
        # dJ/db = 2 * mean(resid), dJ/dm = 2 * mean(resid * x_scaled)
        db = 2.0 * np.mean(resid)
        dm = 2.0 * np.mean(resid * x_scaled)

        # parameter update
        b -= lr * db
        m -= lr * dm

        if verbose and epoch % verbose == 0:
            cost = np.mean(resid ** 2)
            print(f"epoch {epoch:4d} | MSE={cost:.6f} | b={b:.6f} | m={m:.6f}")

    # ----- 4) map (b, m) back to original units -----
    # y_scaled = b + m * x_scaled
    # y = y_min + y_rng * y_scaled
    # x_scaled = (x - x_min) / x_rng
    #
    # => y = y_min + y_rng * (b + m * (x - x_min)/x_rng)
    #    = [y_min + y_rng*b - (y_rng*m/x_rng)*x_min] + [(y_rng*m)/x_rng] * x
    m_orig = (y_rng * m) / x_rng
    b_orig = y_min + y_rng * b - m_orig * x_min

    return b_orig, m_orig


if __name__ == "__main__":
    df = pd.read_csv("home_prices.csv")
    x = df["area_sqr_ft"].to_numpy()
    y = df["price_lakhs"].to_numpy()

    b, m = gradient_descent(x, y, lr=0.1, epochs=3000, verbose=300)
    print(f"\nFinal Results: m={m:.6f}, b={b:.6f}")