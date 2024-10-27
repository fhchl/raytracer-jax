import numpy as np


from PIL import Image
from wakepy import keep

from src.raytracer import Camera, Material, Scene, Sphere, color, point


def main():
    blue_mat = Material(
        albedo=color(0.1, 0.2, 0.5),
    )
    yellow_mat = Material(
        albedo=color(0.8, 0.8, 0.),
    )
    metal_mat = Material(color(0.8, 0.8, 0.8), reflectance=1.0)
    fuzzy_mat = Material(color(0.8, 0.6, 0.2), reflectance=1.0, fuzziness=0.5)
    # dielectric_mat = Material(color(1.0, 1.0, 1.0), 0., 0., 1.5, 1.0)
    scene = Scene(
        objects=[
            Sphere(point(0.0, -100.5, -1), 100.0, yellow_mat),
            Sphere(point(0, 0.0, -1.2), 0.5, blue_mat),
            Sphere(point(-1.0, 0.0, -1.0), 0.5, metal_mat),
            Sphere(point(1.0, 0.0, -1.0), 0.5, fuzzy_mat),
        ]
    )
    camera = Camera(
        center=point(0.0, 0.0, 0.0),
        image_width=400,
        image_height=225,
        focal_length=1.0,
        samples_per_pixel=400,
        sensor_height=2.0,
        max_depth=10,
    )

    with keep.running():
        image = camera.render(scene)

        image = Image.fromarray(np.uint8(255 * image), mode="RGB")
        image.save("rendering.png")
        image.show()


if __name__ == "__main__":
    main()
