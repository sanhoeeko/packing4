import ctypes as ct
from math import pi, sin, cos

import matplotlib.pyplot as plt
import numpy as np

from art import Capsule
from kernel import ker


class MyCapsule:
    def __init__(self, n, d, x, y, theta, color='violet'):
        self.n, self.d = n, d
        self.a, self.b = 1, 1 / (1 + (n - 1) * d / 2)
        self.x, self.y, self.theta = x, y, theta
        self.color = color

    def show(self):
        return Capsule((self.x, self.y), width=np.sqrt(2)*self.a, height=np.sqrt(2)*self.b, angle=180 * self.theta / pi,
                       color=self.color, alpha=0.7)

    def notice(self, x, y):
        if (x - self.x) ** 2 + (y - self.y) ** 2 < self.b ** 2:
            print(f"Noticed. Center at: ({self.x}, {self.y})")
            return True
        return False

    def moveTo(self, x, y):
        self.x, self.y = x, y

    def rotate(self, angle):
        self.theta += angle


class Shape:
    def __init__(self, n, d):
        ker.setRod(n, d)
        ker.dll.interpolateGradient.argtypes = [ct.c_float, ct.c_float, ct.c_float]
        self.func = ker.returnFixedArray(ker.dll.interpolateGradient, 3)

    def force(self, x, y, t):
        """
        return (force: (2,) ndarray, torque: float)
        """
        g = self.func(x, y, t)
        print(g)
        return -np.array(g[0:2]), -g[2]


def rotForce(f: np.ndarray, t) -> np.ndarray:
    U = np.array([[cos(t), -sin(t)], [sin(t), cos(t)]])
    return (U @ f.reshape((2, 1))).reshape(-1)


class ForceInterface:
    def __init__(self, e1: MyCapsule, e2: MyCapsule):
        assert e1.n == e2.n, e1.d == e2.d
        self.shape = Shape(e1.n, e1.d)

    def force(self, e1: MyCapsule, e2: MyCapsule) -> (np.ndarray, np.ndarray):

        def normalize(x: np.ndarray):
            return x / np.linalg.norm(x)

        # a, b = e1.a, e1.b  # suppose that two capsules are identity
        dx = e1.x - e2.x
        dy = e1.y - e2.y
        dt = e1.theta - e2.theta
        dx, dy = rotForce(np.array([dx, dy]), -e2.theta)
        F, moment = self.shape.force(dx, dy, dt)
        if np.linalg.norm(F) == 0:
            return np.zeros((2,)), np.zeros((2,))
        F = rotForce(F, e2.theta)
        u = np.array([cos(e1.theta), sin(e1.theta)])
        sin_phi = np.cross(normalize(F), normalize(u))
        r = moment / (np.linalg.norm(F) * sin_phi)
        center = np.array([e1.x, e1.y])
        force_act_on_point = center + r * u
        return F, force_act_on_point

    def rePlot(self):
        ax.cla()
        for obj in objs:
            ax.add_artist(obj.show())
        # plot force
        force, point = self.force(objs[0], objs[1])
        if np.linalg.norm(force) > 0:
            drawPlacedVector(force, point)
        # set ax range
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)

        plt.draw()


def drawPlacedVector(vector: np.ndarray, pos: np.ndarray):
    # print(f"Draw force: ({pos[0]}, {pos[1]}), ({vector[0]}, {vector[1]})")
    ax.arrow(pos[0], pos[1], vector[0] * 0.1, vector[1] * 0.1, head_width=0.1, head_length=0.1)


class Handler:
    def __init__(self):
        self.current_obj = None

    def notice(self, x, y):
        if self.current_obj is None:
            for obj in objs:
                if obj.notice(x, y):
                    self.current_obj = obj
                    break
        else:
            self.release()

    def release(self):
        # print(f"Placed. Center at: ({self.current_obj.x}, {self.current_obj.y})")
        self.current_obj = None

    def moveTo(self, x, y):
        if self.current_obj is not None:
            self.current_obj.moveTo(x, y)
            fi.rePlot()

    def rotatePlus(self):
        if self.current_obj is not None:
            self.current_obj.rotate(0.1)
            fi.rePlot()

    def rotateMinus(self):
        if self.current_obj is not None:
            self.current_obj.rotate(-0.1)
            fi.rePlot()


handler = Handler()


def on_press(event):
    if event.button == 1:
        handler.notice(event.xdata, event.ydata)


def on_release(event):
    pass


def on_move(event):
    handler.moveTo(event.xdata, event.ydata)


def on_scroll(event):
    if event.button == 'up':
        handler.rotatePlus()
    elif event.button == 'down':
        handler.rotateMinus()


if __name__ == '__main__':
    objs = []
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal', adjustable='box')
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    print("Click and release, then drag capsules. Scroll to spin capsules.")
    objs.append(MyCapsule(2, 0.5, 0, 0, 0, 'springgreen'))
    objs.append(MyCapsule(2, 0.5, 1, 1, 1, 'violet'))
    objs.reverse()
    fi = ForceInterface(*objs)
    fi.rePlot()
    plt.show()
