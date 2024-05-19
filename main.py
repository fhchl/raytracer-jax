from src.raytracer import Camera, Scene, Sphere, point
from PIL import Image
import numpy as np


def main():
    scene = Scene(
        objects=[
            Sphere(point(0.0, -100.5, -1), 100.0),
            Sphere(point(0, 0.0, -1.0), 0.5),
        ]
    )
    camera = Camera(
        center=point(0., 0., 0.),
        image_width=400,
        image_height=225,
        focal_length=1.
    )
    image = camera.render(scene)

    image = Image.fromarray(np.uint8(255 * image), mode="RGB")
    image.save("rendering.png")
    image.show()

if __name__ == "__main__":
    main()
