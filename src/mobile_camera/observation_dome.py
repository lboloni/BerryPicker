#!/usr/bin/env python3
# coding: utf-8

# # Observation dome geometry helpers

"""Geometry helpers for an observation dome."""

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import math


class ObservationDome:
    """
    Represent a hemispherical dome resting on a horizontal plane.

    The dome is modeled as the upper half of a sphere centered at `(x, y, z)` with
    radius `r`. The supporting horizontal plane is therefore located at `z`.
    """

    def __init__(self, x, y, z, r, no_long=10, no_lat=10):
        if r <= 0:
            raise ValueError('r must be positive.')
        if no_long < 1:
            raise ValueError('no_long must be at least 1.')
        if no_lat < 1:
            raise ValueError('no_lat must be at least 1.')
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.no_long = no_long
        self.no_lat = no_lat

    @property
    def plane_z(self):
        """Return the z coordinate of the horizontal plane supporting the dome."""
        return self.z

    def contains_point(self, px, py, pz):
        """
        Return `True` if a point lies inside or on the hemispherical dome.

        Points below the supporting plane are excluded.
        """
        if pz < self.plane_z:
            return False

        dx = px - self.x
        dy = py - self.y
        dz = pz - self.z
        return dx * dx + dy * dy + dz * dz <= self.r * self.r

    def get_lat_long_point(self, lat_index, long_index):
        """
        Return the `(x, y, z)` coordinates of a dome surface grid point.

        `lat_index` ranges from `0` to `no_lat`, where `0` is the top of the dome
        and `no_lat` is the equator. `long_index` ranges from `0` to `no_long - 1`
        and wraps around the dome.
        """
        if not 0 <= lat_index <= self.no_lat:
            raise ValueError(f'lat_index must be between 0 and {self.no_lat}.')
        if not 0 <= long_index < self.no_long:
            raise ValueError(f'long_index must be between 0 and {self.no_long - 1}.')

        polar_angle = (lat_index / self.no_lat) * (math.pi / 2.0)
        azimuth = (long_index / self.no_long) * (2.0 * math.pi)
        radial_xy = self.r * math.sin(polar_angle)

        px = self.x + radial_xy * math.cos(azimuth)
        py = self.y + radial_xy * math.sin(azimuth)
        pz = self.z + self.r * math.cos(polar_angle)
        return (px, py, pz)

    def _normalize_lat_long_points(self, lat_long_points):
        """
        Validate and normalize a subset of dome grid points.

        :param lat_long_points: iterable of `(lat_index, long_index)` pairs or `None`
        :return: list of validated `(lat_index, long_index)` pairs
        """
        if lat_long_points is None:
            return [
                (lat_index, long_index)
                for lat_index in range(self.no_lat + 1)
                for long_index in range(self.no_long)
            ]

        normalized_points = []
        for point in lat_long_points:
            if len(point) != 2:
                raise ValueError(
                    'Each lat/long point must be a `(lat_index, long_index)` pair.'
                )
            lat_index, long_index = point
            self.get_lat_long_point(lat_index, long_index)
            normalized_points.append((lat_index, long_index))
        return normalized_points

    @staticmethod
    def get_ee_pose_components_pointing_at(x, y, z, xv, yv, zv):
        """
        Return a full 6 DoF end-effector pose pointing from `(x, y, z)` to `(xv, yv, zv)`.

        The returned value is a dictionary compatible with
        `bot.arm.set_ee_pose_components(**pose)`. The pointing direction determines
        `pitch` and `yaw`; `roll` is set to `0.0` by convention because the target
        point alone does not constrain rotation about the tool axis.
        """
        dx = xv - x
        dy = yv - y
        dz = zv - z

        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            raise ValueError('The target point must be different from the pose position.')

        horizontal_distance = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx)
        pitch = -math.atan2(dz, horizontal_distance)

        return {
            'x': x,
            'y': y,
            'z': z,
            'roll': 0.0,
            'pitch': pitch,
            'yaw': yaw,
        }

    @staticmethod
    def visualize_ee_pose(ax, x, y, z, roll, pitch, yaw, *, color='tab:red', length=0.05):
        """
        Draw the end-effector pointing direction on an existing 3D Matplotlib axis.

        The arrow starts at `(x, y, z)` and points in the direction implied by
        `pitch` and `yaw`. `roll` is accepted as part of the full 6 DoF pose but
        does not affect the single-arrow visualization.
        """
        if length <= 0:
            raise ValueError('length must be positive.')

        del roll

        dx = length * math.cos(pitch) * math.cos(yaw)
        dy = length * math.cos(pitch) * math.sin(yaw)
        dz = -length * math.sin(pitch)

        return ax.quiver(
            x,
            y,
            z,
            dx,
            dy,
            dz,
            color=color,
            arrow_length_ratio=0.2,
        )

    def visualize(
        self,
        ax,
        *,
        lat_long_points=None,
        pose_target=None,
        title='Observation Dome Grid and End-Effector Poses',
        ring_color='tab:blue',
        arc_color='tab:orange',
        point_color='black',
        center_color='tab:green',
        pose_color='tab:red',
        pose_length=0.12,
    ):
        """
        Draw the dome grid and optional end-effector poses on an existing 3D axis.

        :param ax: Matplotlib 3D axis
        :param lat_long_points: optional iterable of `(lat_index, long_index)` pairs to visualize;
            if `None`, all dome nodes are visualized
        :param pose_target: optional `(x, y, z)` point for pose arrows to point at; if `None`,
            no end-effector arrows are drawn
        :param title: plot title
        :param ring_color: color used for latitude rings
        :param arc_color: color used for longitude arcs
        :param point_color: color used for selected grid points
        :param center_color: color used for the dome center marker
        :param pose_color: color used for end-effector arrows
        :param pose_length: arrow length for end-effector pose visualization
        :return: list of selected `(x, y, z)` points that were visualized
        """
        selected_indices = self._normalize_lat_long_points(lat_long_points)

        for lat_index in range(self.no_lat + 1):
            xs = []
            ys = []
            zs = []
            for long_index in range(self.no_long + 1):
                wrapped_long = long_index % self.no_long
                x, y, z = self.get_lat_long_point(lat_index, wrapped_long)
                xs.append(x)
                ys.append(y)
                zs.append(z)
            ax.plot(xs, ys, zs, color=ring_color, alpha=0.55, linewidth=1.0)

        for long_index in range(self.no_long):
            xs = []
            ys = []
            zs = []
            for lat_index in range(self.no_lat + 1):
                x, y, z = self.get_lat_long_point(lat_index, long_index)
                xs.append(x)
                ys.append(y)
                zs.append(z)
            ax.plot(xs, ys, zs, color=arc_color, alpha=0.55, linewidth=1.0)

        points = [
            self.get_lat_long_point(lat_index, long_index)
            for lat_index, long_index in selected_indices
        ]
        if points:
            xs, ys, zs = zip(*points)
            ax.scatter(xs, ys, zs, color=point_color, s=24)
        ax.scatter([self.x], [self.y], [self.z], color=center_color, s=60)

        if pose_target is not None:
            xv, yv, zv = pose_target
            for x, y, z in points:
                pose = self.get_ee_pose_components_pointing_at(x, y, z, xv, yv, zv)
                self.visualize_ee_pose(
                    ax,
                    pose['x'],
                    pose['y'],
                    pose['z'],
                    pose['roll'],
                    pose['pitch'],
                    pose['yaw'],
                    color=pose_color,
                    length=pose_length,
                )

        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect((1, 1, 0.7))
        ax.set_xlim(self.x - self.r, self.x + self.r)
        ax.set_ylim(self.y - self.r, self.y + self.r)
        ax.set_zlim(self.z, self.z + self.r)

        return points
