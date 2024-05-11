from PIL import Image
import numpy as np
from tqdm import tqdm
from .geometry import (
    Sphere,
    point,
    Scene,
    vector,
    Color,
    Ray,
    color,
    unit_vector,
    Interval,
)
import jax
import jax.numpy as jnp
from jaxtyping import Scalar

# Image
aspect_ratio = 16.0 / 9.0
image_width = 400
image_height = int(image_width / aspect_ratio)

# Scene
scene = Scene(
    objects=[
        Sphere(point(0.0, -100.5, -1), 100.0),
        Sphere(point(0, 0.0, -1.0), 0.5),
    ]
)

# Camera
focal_length = 1.0
viewport_height = 2.0
viewport_width = viewport_height * (image_width / image_height)
camera_center = vector(0.0, 0.0, 0.0)

# Viewport
viewport_u = vector(viewport_width, 0.0, 0.0)
viewport_v = vector(0.0, -viewport_height, 0.0)
pixel_delta_u = viewport_u / image_width
pixel_delta_v = viewport_v / image_height
viewport_upper_left = (
    camera_center - vector(0.0, 0.0, focal_length) - viewport_u / 2 - viewport_v / 2
)
pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)


def ray_color(r: Ray) -> Color:
    rec = scene.hit(r, Interval(0.0, jnp.inf))

    def hit():
        return 0.5 * (rec.normal + 1)

    def nohit():
        unit_direction = unit_vector(r.direction)
        a = 0.5 * (unit_direction[1] + 1)
        return (1 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0)

    is_hit = Interval(0., jnp.inf).contains(rec.t)
    return jax.lax.cond(is_hit, hit, nohit)


def compute_pixel(i: Scalar, j: Scalar) -> Color:
    pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v)
    ray_direction = pixel_center - camera_center
    r = Ray(camera_center, ray_direction)
    return ray_color(r)


jit = jax.jit(compute_pixel)


def main() -> None:
    image = np.zeros((image_height, image_width, 3))
    for j in tqdm(range(image_height)):
        for i in range(image_width):
            image[j, i, :] = jit(i, j)

    image = Image.fromarray(np.uint8(255 * image), mode="RGB")
    image.save("rendering.png")
    image.show()
