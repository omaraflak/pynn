"""Canvas class for testing MNIST model."""

from pynn import modules
import numpy as np
import matplotlib.pyplot as plt
import cv2


class MnistCanvas:
    """Canvas class for testing MNIST model."""

    def __init__(self, model: modules.Module):
        self.x_coords = []
        self.y_coords = []
        self.model = model
        self.size = 28

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        self.ax.set_aspect("equal", adjustable="box")

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

    def show(self):
        plt.show()

    def _on_click(self, event):
        if event.inaxes:
            self.x_coords.append(event.xdata)
            self.y_coords.append(event.ydata)
            plt.plot(self.x_coords, self.y_coords, "k-", lw=30)
            plt.draw()

    def _on_move(self, event):
        if event.inaxes and event.button == 1:
            self.x_coords.append(event.xdata)
            self.y_coords.append(event.ydata)
            plt.plot(self.x_coords, self.y_coords, "k-", lw=30)
            plt.draw()

    def _on_release(self, _):
        self.x_coords = []
        self.y_coords = []

    def _on_key_press(self, event):
        if event.key == "c":
            self.ax.cla()
            self.x_coords = []
            self.y_coords = []
            self.ax.set_xlim(0, self.size)
            self.ax.set_ylim(0, self.size)
            plt.draw()
        elif event.key == "v":
            self.fig.canvas.draw()
            output = self.model.forward(self._get_image())
            idx = np.argmax(output)
            print(
                f"Predicted class: {idx}, probability: {int(output[idx] * 100)}%")

    def _get_image(self) -> np.ndarray:
        path = "/tmp/img.png"
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        new_image = 255 - image
        new_image = cv2.resize(new_image, (self.size, self.size))
        cx, cy = self._get_center_of_gravity(new_image)
        new_image = self._center_image(new_image, cx, cy)
        cv2.imwrite(path, new_image)
        new_image = np.reshape(new_image, (self.size * self.size, 1))
        new_image = new_image.astype(np.float32) / 255
        return new_image

    def _get_center_of_gravity(self, image: np.ndarray) -> tuple[int, int]:
        rows, cols = np.where(image > 20)
        return int(np.mean(cols)), int(np.mean(rows))

    def _center_image(self, image: np.ndarray, x: int, y: int):
        height, width = image.shape
        new_image = np.zeros_like(image)

        # Calculate the offsets for the image
        x_offset = int(width / 2 - x)
        y_offset = int(height / 2 - y)

        # Calculate the slicing indices for the original and new images
        x_start_old = max(0, -x_offset)
        x_end_old = min(width, width - x_offset)
        y_start_old = max(0, -y_offset)
        y_end_old = min(height, height - y_offset)

        x_start_new = max(0, x_offset)
        x_end_new = min(width, width + x_offset)
        y_start_new = max(0, y_offset)
        y_end_new = min(height, height + y_offset)

        # Copy the relevant part of the image to the new image
        new_image[y_start_new:y_end_new, x_start_new:x_end_new] = image[
            y_start_old:y_end_old, x_start_old:x_end_old
        ]

        return new_image
