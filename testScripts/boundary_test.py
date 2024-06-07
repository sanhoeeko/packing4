import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from kernel_for_test import ker


class DraggablePoint:
    lock = None

    def __init__(self, point, line):
        self.point = point
        self.line = line
        self.theta = 0
        self.press = None
        self.background = None
        xp, yp, tp = self.calMirror()
        self.mirror_point = plt.Circle((xp, yp), 0.2, color='orange')
        self.mirror_line, = ax.plot((xp, yp), (px + np.cos(tp), py + np.sin(tp)), 'r-', color='blue')
        ax.add_patch(self.point)
        ax.add_patch(self.mirror_point)

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidscroll = self.point.figure.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_press(self, event):
        if event.inaxes != self.point.axes: return
        contains, attrd = self.point.contains(event)
        if not contains: return
        self.press = (self.point.center), event.xdata, event.ydata
        DraggablePoint.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.point.figure.canvas
        axes = self.point.axes
        self.point.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.point.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.point)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        if DraggablePoint.lock is not self: return
        if event.inaxes != self.point.axes: return
        self.point.center, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.point.center = (self.point.center[0] + dx, self.point.center[1] + dy)

        # update self.line
        x = np.array([-100, 100])  # large enough to cover the plot area
        y = np.tan(self.theta) * (x - self.point.center[0]) + self.point.center[1]
        self.line.set_data(x, y)

        # update mirror point and line
        xp, yp, tp = self.calMirror()
        self.mirror_point.center = (xp, yp)
        y_mirror = np.tan(tp) * (x - xp) + yp
        self.mirror_line.set_data(x, y_mirror)

        canvas = self.point.figure.canvas
        axes = self.point.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.point)
        axes.draw_artist(self.line)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        'on release we reset the press data'
        if DraggablePoint.lock is not self: return

        self.press = None
        DraggablePoint.lock = None

        # turn off the rect animation property and reset the background
        self.point.set_animated(False)
        self.background = None

        # redraw the full figure
        self.point.figure.canvas.draw()

    def on_scroll(self, event):
        'rotate self.line on scroll'
        if event.inaxes != self.point.axes: return
        self.theta += np.deg2rad(5) if event.button == 'up' else np.deg2rad(-5)
        x = np.array([-100, 100])  # large enough to cover the plot area
        y = np.tan(self.theta) * (x - self.point.center[0]) + self.point.center[1]
        self.line.set_data(x, y)

        # update mirror point and line
        xp, yp, tp = self.calMirror()
        self.mirror_point.center = (xp, yp)
        y_mirror = np.tan(tp) * (x - xp) + yp
        self.mirror_line.set_data(x, y_mirror)

        self.line.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)
        self.point.figure.canvas.mpl_disconnect(self.cidscroll)

    def getXyt(self):
        'get the x, y position and theta (rotation angle) of the point'
        return self.point.center[0], self.point.center[1], self.theta

    def calMirror(self):
        'invoke the kernel to calculate the mirror point / orientation'
        x, y, t = self.getXyt()
        xp, yp, tp = ker.getMirrorOf(A, B, x, y, t)
        if xp == 0 and yp == 0 and tp == 0:
            print("Invalid collision")
            return np.nan, np.nan, np.nan
        else:
            print(xp, yp, tp)
            return xp, yp, tp


fig, ax = plt.subplots()

# draw the boundary
A, B = 24, 12
x0, x1 = -26, -18
y0, y1 = -6, 2
ellipse = patches.Ellipse((0, 0), width=2 * 24, height=2 * 12, fill=False)
ax.add_artist(ellipse)
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect('equal')

# draw a point and a line
px, py = (x0 + x1) / 2, (y0 + y1) / 2
point = plt.Circle((px, py), 0.2)
line, = ax.plot((px, py), (px + 1, py + 1), 'r-')

draggable_point = DraggablePoint(point, line)
draggable_point.connect()

plt.show()
