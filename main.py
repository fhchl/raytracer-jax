import numpy as np
from PIL import Image

from src.raytracer import Camera, Scene, Sphere, point


def main():
    scene = Scene(
        objects=[
            Sphere(point(0.0, -100.5, -1), 100.0),
            Sphere(point(0, 0.0, -1.0), 0.5),
        ]
    )
    camera = Camera(
        center=point(0.0, 0.0, 0.0),
        image_width=400,
        image_height=225,
        focal_length=1.0,
        samples_per_pixel=100,
        sensor_height=2.0,
    )
    image = camera.render(scene)

    print(np.max(image), np.min(image))
    image = Image.fromarray(np.uint8(255 * image), mode="RGB")
    image.save("rendering.png")
    image.show()


if __name__ == "__main__":
    main()
