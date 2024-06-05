import matplotlib.pyplot as plt

from myio import DataSet
from render import StateRenderer


class InteractiveViewer:
    def __init__(self, dataset: DataSet):
        self.metadata = dataset.metadata
        self.data = list(map(lambda x: StateRenderer(x), dataset.data))
        self.index = 0
        self.handle = plt.subplots()
        self.fig, self.ax = self.handle
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_key_press(self, event):
        if event.key == 'left':
            self.index = max(0, self.index - 1)
        elif event.key == 'right':
            self.index = min(len(self.data) - 1, self.index + 1)
        else:
            return

        self.ax.clear()
        self.data[self.index].drawBoundary(self.handle)
        self.data[self.index].drawParticles(self.handle)
        plt.draw()

    def show(self):
        self.data[self.index].drawBoundary(self.handle)
        self.data[self.index].drawParticles(self.handle)
        plt.show()


ds = DataSet.loadFrom('data.h5')
iv = InteractiveViewer(ds)
iv.show()
