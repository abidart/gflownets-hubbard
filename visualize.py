from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, Rectangle
import matplotlib.pyplot as pp
import numpy as np
from checks import check_valid_lattice

x1 = 0.35
x2 = 0.65
pad1 = 0.2
pad2 = 0.2
line_pad = 0.2
particle_separation = 0.3


def _base_lattice(lattice_width, lattice_height):
    return tuple(
        [
            pp.gca().add_patch(
                pp.Circle(
                    (x * particle_separation, y * particle_separation),
                    0.05,
                    fc=(0.9, 0.9, 0.9),
                    zorder=4,
                )
            )
            for y in range(lattice_width)
            for x in range(lattice_height)
        ]
        + [
            pp.gca().add_line(
                Line2D(
                    [-line_pad, (lattice_height - 1) * particle_separation + line_pad],
                    [x * particle_separation, x * particle_separation],
                    color="grey",
                    zorder=1,
                )
            )
            for x in range(lattice_width)
        ]
        + [
            pp.gca().add_line(
                Line2D(
                    [x * particle_separation, x * particle_separation],
                    [-line_pad, (lattice_width - 1) * particle_separation + line_pad],
                    color="grey",
                    zorder=1,
                )
            )
            for x in range(lattice_height)
        ]
        + [
            # Add a square around the image
            pp.gca().add_patch(
                Rectangle(
                    (0 - line_pad, 0 - line_pad),
                    line_pad * 2 + (lattice_height - 1) * particle_separation,
                    line_pad * 2 + (lattice_width - 1) * particle_separation,
                    fill=None,
                    edgecolor="black",
                    zorder=0,
                    lw=2,
                )
            )
        ]
    )


def _draw_arrows(spin, lattice):
    width, height = lattice.shape
    for x in range(width):
        for y in range(height):
            if lattice[x][y] == 1:
                if spin == 0:
                    pp.gca().add_patch(
                        FancyArrow(
                            y * particle_separation,
                            (width - 1 - x) * particle_separation,
                            0,
                            -0.1,
                            width=0.020,
                            head_width=0.04,
                            head_length=0.020,
                            fc="red",
                            ec="red",
                            zorder=3,
                        )
                    )
                else:  # spin equals 1
                    pp.gca().add_patch(
                        FancyArrow(
                            y * particle_separation,
                            (width - 1 - x) * particle_separation,
                            0,
                            0.1,
                            width=0.020,
                            head_width=0.04,
                            head_length=0.020,
                            fc="blue",
                            ec="blue",
                            zorder=3,
                        )
                    )


def draw_lattice(lattice):
    check_valid_lattice(example_lattice_1)
    width, height = lattice.shape[1], lattice.shape[2]
    _base_lattice(lattice_width=width, lattice_height=height)
    for spin in [0, 1]:
        spin_lattice = lattice[spin]
        _draw_arrows(spin=spin, lattice=spin_lattice)
    pp.axis("scaled")
    pp.axis("off")


if __name__ == "__main__":
    example_lattice_1 = np.array(
        [
            [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 0, 0, 0, 1]],
            [[1, 0, 1, 1, 1], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]],
        ]
    )
    draw_lattice(example_lattice_1)
    pp.show()
