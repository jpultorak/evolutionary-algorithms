import numpy as np


class MLPPolicy:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w1_shape = (input_size, hidden_size)
        self.b1_shape = (hidden_size,)
        self.w2_shape = (hidden_size, output_size)
        self.b2_shape = (output_size,)

        self.param_count = (
            (input_size * hidden_size)
            + hidden_size
            + (hidden_size * output_size)
            + output_size
        )

    def get_action(self, x: np.ndarray, weights_flat: np.ndarray) -> np.ndarray:
        end_w1 = self.input_size * self.hidden_size
        w1 = weights_flat[:end_w1].reshape(self.w1_shape)

        end_b1 = end_w1 + self.hidden_size
        b1 = weights_flat[end_w1:end_b1]

        end_w2 = end_b1 + (self.hidden_size * self.output_size)
        w2 = weights_flat[end_b1:end_w2].reshape(self.w2_shape)

        b2 = weights_flat[end_w2:]

        x = np.tanh(np.dot(x, w1) + b1)
        x = np.tanh(np.dot(x, w2) + b2)

        return x

    def __repr__(self):
        return (
            f"MLPPolicy(in={self.input_size}, hidden={self.hidden_size},"
            f" out={self.output_size}, params={self.param_count})"
        )
