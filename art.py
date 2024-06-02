import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.transforms import Affine2D


class Capsule(patches.Patch):
    def __init__(self, xy, width, height, angle=0, **kwargs):
        """
        :param xy: The center of the capsule.
        :param width: The total width (diameter) of the capsule (including the line and semi-circles).
        :param height: The height (length) of the capsule.
        :param angle: The angle of rotation of the capsule.
        """
        self.xy = xy
        self.width = width
        self.height = height
        self.angle = angle
        super().__init__(**kwargs)

    def get_path(self):
        """
        Get the path of the capsule (a line with semi-circles at the ends).
        """
        # Create the path for the capsule
        c = self.width - self.height
        capsule_path = Path.make_compound_path(
            Path.unit_rectangle().transformed(
                Affine2D().scale(c, self.height).translate(-c / 2, -self.height / 2)),
            Path.unit_circle().transformed(Affine2D().scale(self.height / 2, self.height / 2).translate(-c / 2, 0)),
            Path.unit_circle().transformed(Affine2D().scale(self.height / 2, self.height / 2).translate(c / 2, 0))
        )

        return capsule_path

    def get_patch_transform(self):
        """
        Get the transformation for the capsule.
        """
        # Scale and rotate the capsule
        scale = Affine2D().scale(self.width, self.width).rotate_deg(self.angle)

        # Translate the capsule to the correct location
        trans = Affine2D().translate(*self.xy)

        return scale + trans
