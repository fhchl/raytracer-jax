from jax import Array
import jax.numpy as jnp
import equinox as eqx
import jax
import jax.tree_util as jtu
from tqdm import tqdm
from .util import tree_stack
import numpy as np
from beartype.claw import beartype_this_package
from jaxtyping import ScalarLike, Scalar

beartype_this_package()

Bool = Float = Color = Point = Vector = Array
color = point = vector = lambda x, y, z: jnp.array((x, y, z))


def unit_vector(v: Array) -> Array:
    return v / length(v)


def length_squared(v: Array) -> Float:
    return v.dot(v)


def length(v: Array) -> Float:
    return jnp.sqrt(length_squared(v))


def intersect_line_plane(
    plane_origin: Point,
    plane_normal: Vector,
    line_origin: Point,
    line_direction: Vector,
) -> Float:
    return (plane_origin - line_origin).dot(plane_normal) / line_direction.dot(
        plane_normal
    )


def intersect_line_sphere(
    center: Point, radius: Float, origin: Point, direction: Vector
) -> Float:
    # Closest point on line to center is q
    t = intersect_line_plane(center, direction, origin, direction)
    q = direction * t + origin
    # Distance q to c
    d_squared = length_squared(center - q)
    return jnp.where(d_squared < radius**2, t - jnp.sqrt(radius**2 - d_squared), jnp.inf)


class Ray(eqx.Module):
    origin: Point
    direction: Vector

    def __post_init__(self):
        self.direction = unit_vector(self.direction)

    def at(self, t: Float) -> Point:
        return self.origin + t * self.direction


class HitRecord(eqx.Module):
    p: Point
    normal: Vector
    t: Float
    front_face: Bool

    def set_face_normal(self, r: Ray, outward_normal: Vector) -> "HitRecord":
        front_face = jnp.where(r.direction.dot(outward_normal) > 0, True, False)
        normal = jnp.where(front_face, outward_normal, -outward_normal)
        return HitRecord(self.p, normal, self.t, front_face)


class Interval(eqx.Module):
    min: float = jnp.inf
    max: float = -jnp.inf

    def size(self) -> float:
        return self.max - self.min

    def contains(self, x: ScalarLike) -> Bool:
        x = jnp.asarray(x)
        return (self.min < x) & (x < self.max)

    def surrounds(self, x: ScalarLike) -> Bool:
        x = jnp.asarray(x)
        return (self.min <= x) & (x <= self.max)


class Sphere(eqx.Module):
    center: Vector
    radius: float

    def hit(self, r: Ray, ray_t: Interval) -> HitRecord:
        # Compute closest point Q to sphere
        t = intersect_line_sphere(self.center, self.radius, r.origin, r.direction)
        def hit():
            p = r.at(t)
            normal = (p - self.center) / self.radius
            return HitRecord(p, normal, t, True)
        def nohit():
            return HitRecord(
                point(0.0, 0.0, 0.0), vector(0.0, 0.0, 0.0), jnp.inf, False
            )
        is_hit = ray_t.contains(t)
        return jax.lax.cond(is_hit, hit, nohit)


class Scene(eqx.Module):
    objects: list[Sphere]

    def hit(self, r: Ray, ray_t: Interval) -> HitRecord:
        stacked = tree_stack(self.objects)
        hits = jax.vmap(lambda obj: obj.hit(r, ray_t))(stacked)
        idx = jnp.argmin(hits.t, axis=0)
        return jtu.tree_map(lambda n: n[idx], hits)

class Camera(eqx.Module):
    center: Point
    image_width: int
    image_height: int
    focal_length: float

    
    def render(self, scene: Scene) -> np.ndarray:        
        # Camera
        viewport_height = 2.0
        viewport_width = viewport_height * (self.image_width / self.image_height)

        # Viewport
        viewport_u = vector(viewport_width, 0.0, 0.0)
        viewport_v = vector(0.0, -viewport_height, 0.0)
        pixel_delta_u = viewport_u / self.image_width
        pixel_delta_v = viewport_v / self.image_height
        viewport_upper_left = (
            self.center - vector(0.0, 0.0, self.focal_length) - viewport_u / 2 - viewport_v / 2
        )
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)

        def compute_pixel(i: Scalar, j: Scalar) -> Color:
            pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v)
            ray_direction = pixel_center - self.center
            r = Ray(self.center, ray_direction)
            return self._ray_color(r, scene)

        jit = jax.jit(compute_pixel)

        # TODO: do some vmap here
        image = np.zeros((self.image_height, self.image_width, 3))
        for j in tqdm(range(self.image_height)):
            for i in range(self.image_width):
                image[j, i, :] = jit(i, j)

        return image

    def _ray_color(self, r: Ray, scene: Scene) -> Color:    
        rec = scene.hit(r, Interval(0.0, jnp.inf))

        def hit():
            return 0.5 * (rec.normal + 1)

        def nohit():
            unit_direction = unit_vector(r.direction)
            a = 0.5 * (unit_direction[1] + 1)
            return (1 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0)

        is_hit = Interval(0., jnp.inf).contains(rec.t)
        return jax.lax.cond(is_hit, hit, nohit)
        
    
