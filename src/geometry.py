from jax import Array
import jax.numpy as jnp
import equinox as eqx
import jax
import jax.tree_util as jtu

from .util import tree_stack

from beartype.claw import beartype_this_package

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


class Sphere(eqx.Module):
    center: Vector
    radius: float

    def hit(self, r: Ray, ray_tmin: float, ray_tmax: float) -> HitRecord:
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
        is_hit = jnp.logical_and(0 < t, t < jnp.inf)
        return jax.lax.cond(is_hit, hit, nohit)


class Scene(eqx.Module):
    objects: list[Sphere]

    def hit(self, r: Ray, ray_tmin: float, ray_tmax: float) -> HitRecord:
        stacked = tree_stack(self.objects)
        hits = jax.vmap(lambda obj: obj.hit(r, ray_tmin, ray_tmax))(stacked)
        idx = jnp.argmin(hits.t, axis=0)
        return jtu.tree_map(lambda n: n[idx], hits)
