from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import jaxtyping as jt
from jaxtyping import ScalarLike
from tqdm import tqdm
from typing import TypeAlias

from .util import tree_stack

Int: TypeAlias = jt.Int[jt.Array, ""]
Bool: TypeAlias = jt.Bool[jt.Array, ""]
Float: TypeAlias = jt.Float[jt.Array, ""]
Color: TypeAlias = jt.Float[jt.Array, "3"]
Point: TypeAlias = jt.Float[jt.Array, "3"]
Vector: TypeAlias = jt.Float[jt.Array, "3"]
KeyArray: TypeAlias = jt.Key[jt.Array, "..."]

color = point = vector = lambda x, y, z: jnp.array((x, y, z))


def unit_vector(v: Vector) -> Vector:
    return v / length(v)


def random_on_hemisphere(normal: Vector, key: KeyArray) -> Vector:
    on_unit_sphere = jax.random.ball(key, 3)
    return on_unit_sphere * jnp.sign(on_unit_sphere.dot(normal))


def length_squared(v: Vector) -> Float:
    return v.dot(v)


def length(v: Vector) -> Float:
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
    return jnp.where(
        d_squared < radius**2, t - jnp.sqrt(radius**2 - d_squared), jnp.inf
    )


def linear_to_gamma(c: np.ndarray) -> np.ndarray:
    return np.where(c > 0, np.sqrt(c), 0)


def reflect(v: Vector, n: Vector) -> Vector:
    return v - 2 * v.dot(n) * n


def refract(uv: Vector, n: Vector, etai_over_etat: Float):
    r_out_perp = etai_over_etat * (uv - uv.dot(n) * n)
    r_out_para = -jnp.sqrt(1.0 - length_squared(r_out_perp)) * n
    return r_out_perp + r_out_para


class Ray(eqx.Module):
    origin: Point
    direction: Vector

    def __post_init__(self):
        self.direction = unit_vector(self.direction)

    def at(self, t: Float) -> Point:
        return self.origin + t * self.direction


class Material(eqx.Module):
    albedo: Color
    reflectance: float = 0.0
    fuzziness: float = 0.0
    refraction_index: float = 1.0
    transparency: float = 0.0

    def scatter(self, r_in: Ray, rec: HitRecord, key: KeyArray) -> tuple[Color, Ray]:
        key1, key2, key3 = jax.random.split(key, 3)

        def reflection():
            lambertian = rec.normal + jax.random.ball(key1, 3)
            reflected = reflect(r_in.direction, rec.normal)
            metal = unit_vector(reflected) + self.fuzziness * jax.random.ball(key2, 3)
            direction = (1 - self.reflectance) * lambertian + self.reflectance * metal
            return direction

        def transmission():
            ri = jax.lax.select(
                rec.front_face,
                1.0 / self.refraction_index,
                self.refraction_index,
            )
            unit_direction = unit_vector(r_in.direction)
            direction = refract(unit_direction, rec.normal, ri)
            return direction

        is_reflected = jax.random.uniform(key3) > self.transparency
        direction = jax.lax.cond(is_reflected, reflection, transmission)

        return self.albedo, Ray(rec.p, direction)


class HitRecord(eqx.Module):
    p: Point
    normal: Vector
    t: Float
    front_face: Bool
    mat: Material


def compute_normal_and_face(r: Ray, outward_normal: Vector) -> tuple[Vector, Bool]:
    front_face = r.direction.dot(outward_normal) < 0
    normal = jnp.where(front_face, outward_normal, -outward_normal)
    return normal, front_face


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

    def clamp(self, x: ScalarLike):
        return jax.lax.clamp(self.min, x, self.max)


class Sphere(eqx.Module):
    center: Vector
    radius: Float
    mat: Material

    def hit(self, r: Ray, ray_t: Interval) -> HitRecord:
        # Compute closest point Q to sphere
        t = intersect_line_sphere(self.center, self.radius, r.origin, r.direction)

        def hit():
            p = r.at(t)
            outward_normal = (p - self.center) / self.radius
            normal, front_face = compute_normal_and_face(r, outward_normal)
            return HitRecord(p, normal, t, front_face, self.mat)

        def nohit():
            null = point(0.0, 0.0, 0.0)
            return HitRecord(null, null, jnp.array(jnp.inf), jnp.array(False), self.mat)

        is_hit = ray_t.contains(t)
        return jax.lax.cond(is_hit, hit, nohit)


class Scene(eqx.Module):
    objects: list[Sphere]

    def hit(self, r: Ray, ray_t: Interval) -> HitRecord:
        stacked = tree_stack(self.objects)
        hits = jax.vmap(lambda obj: obj.hit(r, ray_t))(stacked)
        idx = jnp.argmin(hits.t, axis=0)
        return jtu.tree_map(lambda n: n[idx], hits)


def _sample_square(key: KeyArray) -> jt.Float[jt.Array, "2"]:
    return jax.random.uniform(key=key, shape=(2,), minval=-0.5, maxval=0.5)


class MarchingStep(eqx.Module):
    ray: Ray
    attenuation: Color
    rec: HitRecord
    depth: int
    key: KeyArray


class Camera(eqx.Module):
    center: Point
    image_width: int
    image_height: int
    focal_length: float
    samples_per_pixel: int
    sensor_height: float
    max_depth: int

    def render(self, scene: Scene, seed: int = 0) -> np.ndarray:
        def sample_pixel(i: ScalarLike, j: ScalarLike, key: KeyArray) -> Color:
            key, subkey = jax.random.split(key)
            r = self._get_ray(i, j, key)
            return self._ray_color(r, scene, subkey)

        def compute_pixel(i: Int, j: Int, key: KeyArray) -> Color:
            keys = jax.random.split(key, self.samples_per_pixel)
            colors = jax.vmap(sample_pixel, in_axes=(None, None, 0))(i, j, keys)
            return jnp.mean(colors, axis=0)

        # Don't use decorator here to avoid beartype warning
        jit_compute_pixel = jax.jit(jax.vmap(compute_pixel, in_axes=(0, None, 0)))

        key = jax.random.key(seed)
        image_shape = (self.image_height, self.image_width)
        image = np.zeros((*image_shape, 3))
        keys = jax.random.split(key, image_shape)
        i = np.arange(self.image_width)
        for j in tqdm(range(self.image_height)):
            image[j, i, :] = jit_compute_pixel(i, j, keys[j])

        # FIXME: without linear_to_gamma, sky looks good but yellow not. Is something
        #        off with how we save colors?
        image = np.clip(linear_to_gamma(image), 0.0, 0.999)

        return image

    def _get_ray(self, i: ScalarLike, j: ScalarLike, key: KeyArray) -> Ray:
        sensor_width = self.sensor_height * (self.image_width / self.image_height)
        viewport_u = vector(sensor_width, 0.0, 0.0)
        viewport_v = vector(0.0, -self.sensor_height, 0.0)
        pixel_delta_u = viewport_u / self.image_width
        pixel_delta_v = viewport_v / self.image_height
        viewport_upper_left = (
            self.center
            - vector(0.0, 0.0, self.focal_length)
            - viewport_u / 2
            - viewport_v / 2
        )
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)
        offset = _sample_square(key)
        pixel_sample = (
            pixel00_loc
            + ((i + offset[0]) * pixel_delta_u)
            + ((j + offset[1]) * pixel_delta_v)
        )
        ray_direction = pixel_sample - self.center
        return Ray(self.center, ray_direction)

    def _ray_color(self, r: Ray, scene: Scene, key: KeyArray) -> Color:
        def is_hit(step: MarchingStep) -> Bool:
            return (step.depth > 0) & Interval(0.0, jnp.inf).contains(step.rec.t)

        def march(step: MarchingStep) -> MarchingStep:
            key, new_key = jax.random.split(step.key)
            attenuation, new_ray = step.rec.mat.scatter(step.ray, step.rec, key)
            new_hit = scene.hit(new_ray, Interval(0.001, jnp.inf))
            new_depth = step.depth - 1
            new_attenuation = jax.lax.select(
                new_depth > 0, attenuation * step.attenuation, color(0.0, 0.0, 0.0)
            )
            return MarchingStep(new_ray, new_attenuation, new_hit, new_depth, new_key)

        # March until no object is hit
        initial_step = MarchingStep(
            ray=r,
            attenuation=color(1.0, 1.0, 1.0),
            rec=scene.hit(r, Interval(0.0, jnp.inf)),
            depth=self.max_depth,
            key=key,
        )
        last_step = jax.lax.while_loop(is_hit, march, initial_step)

        # Finally hit the sky
        unit_direction = unit_vector(last_step.ray.direction)
        a = 0.5 * (unit_direction[1] + 1)
        return last_step.attenuation * (
            (1 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0)
        )
