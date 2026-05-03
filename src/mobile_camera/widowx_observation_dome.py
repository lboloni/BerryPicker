#!/usr/bin/env python3
# coding: utf-8

# # WidowX observation dome driver

"""Helpers for driving a WidowX arm over an observation dome."""

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"


class WidowX_ObservationDomeDriver:
    """
    Drive a WidowX arm to latitude/longitude nodes on an `ObservationDome`.

    The driver assumes the arm interface supports `set_ee_pose_components(...)`, such as
    `InterbotixManipulatorXS.arm`.
    """

    def __init__(self, arm, dome, safety_box=None):
        self.arm = arm
        self.dome = dome
        if safety_box is None:
            safety_box = {
                'x': (0.0, 0.40),
                'y': (-0.20, 0.20),
                'z': (0.0, 0.40),
            }
        self.safety_box = safety_box
        self.reached_pose_history = []
        self.current_lat_index = None
        self.current_long_index = None

    def set_safety_box(self, safety_box):
        """
        Set or replace the rectangular safety box.

        :param safety_box: `None` to disable bounds checking, or a dictionary like
            `{'x': (xmin, xmax), 'y': (ymin, ymax), 'z': (zmin, zmax)}`
        """
        self.safety_box = safety_box

    def clear_reached_pose_history(self):
        """Clear the recorded history of successfully reached end-effector poses."""
        self.reached_pose_history = []

    def get_current_lat_long(self):
        """Return the current `(lat_index, long_index)` position, if known."""
        if self.current_lat_index is None or self.current_long_index is None:
            return None
        return self.current_lat_index, self.current_long_index

    def clear_current_lat_long(self):
        """Clear the remembered latitude/longitude position."""
        self.current_lat_index = None
        self.current_long_index = None

    def _is_current_lat_long(self, lat_index, long_index):
        """Return `True` if the driver already believes it is at this dome node."""
        return (
            self.current_lat_index == lat_index
            and self.current_long_index == long_index
        )

    def _record_current_lat_long(self, lat_index, long_index):
        """Record the current latitude/longitude position."""
        self.current_lat_index = lat_index
        self.current_long_index = long_index

    def _record_reached_pose(self, command_pose):
        """Record a successfully reached end-effector pose."""
        self.reached_pose_history.append(dict(command_pose))

    def _plot_safety_box(
        self,
        ax,
        *,
        color='tab:purple',
        alpha=0.35,
        linewidth=1.2,
    ):
        """Draw the safety box on an existing 3D Matplotlib axis."""
        if self.safety_box is None:
            return

        x0, x1 = self.safety_box['x']
        y0, y1 = self.safety_box['y']
        z0, z1 = self.safety_box['z']
        corners = [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for start_index, end_index in edges:
            xs = [corners[start_index][0], corners[end_index][0]]
            ys = [corners[start_index][1], corners[end_index][1]]
            zs = [corners[start_index][2], corners[end_index][2]]
            ax.plot(xs, ys, zs, color=color, alpha=alpha, linewidth=linewidth)

    def visualize_history(
        self,
        ax,
        *,
        show_dome=True,
        show_safety_box=True,
        plot_positions_only=False,
        pose_target=None,
        dome_kwargs=None,
        safety_box_color='tab:purple',
        history_color='tab:red',
        history_marker_color='tab:red',
        history_marker_size=28,
        history_linewidth=1.8,
        title='Observation Dome, Safety Box, and Reached Pose History',
    ):
        """
        Visualize the dome, safety box, and reached end-effector pose history on a 3D axis.

        :param ax: Matplotlib 3D axis
        :param show_dome: if `True`, draw the dome visualization as context
        :param show_safety_box: if `True`, draw the safety box
        :param plot_positions_only: if `True`, visualize only XYZ positions and suppress dome
            pose arrows
        :param pose_target: optional target used when drawing dome pose arrows
        :param dome_kwargs: optional extra keyword arguments forwarded to `dome.visualize(...)`
        """
        if dome_kwargs is None:
            dome_kwargs = {}

        if show_dome:
            dome_plot_kwargs = dict(dome_kwargs)
            if plot_positions_only:
                dome_plot_kwargs.pop('pose_target', None)
            elif pose_target is not None:
                dome_plot_kwargs.setdefault('pose_target', pose_target)
            self.dome.visualize(ax, **dome_plot_kwargs)

        if show_safety_box:
            self._plot_safety_box(ax, color=safety_box_color)

        if self.reached_pose_history:
            xs = [pose['x'] for pose in self.reached_pose_history]
            ys = [pose['y'] for pose in self.reached_pose_history]
            zs = [pose['z'] for pose in self.reached_pose_history]
            ax.plot(
                xs,
                ys,
                zs,
                color=history_color,
                linewidth=history_linewidth,
            )
            ax.scatter(
                xs,
                ys,
                zs,
                color=history_marker_color,
                s=history_marker_size,
            )

        ax.set_title(title)

    def is_pose_safe(self, x, y, z):
        """
        Return `True` if an end-effector position lies inside the configured safety box.

        If no safety box is configured, all positions are considered safe.
        """
        if self.safety_box is None:
            return True

        for axis_name, axis_value in (('x', x), ('y', y), ('z', z)):
            if axis_name not in self.safety_box:
                raise ValueError(
                    f"Safety box is missing bounds for axis '{axis_name}'."
                )
            lower, upper = self.safety_box[axis_name]
            if axis_value < lower or axis_value > upper:
                return False
        return True

    def safe_set_ee_pose_components(self, **kwargs):
        """
        Safely call `set_ee_pose_components(...)` after checking the safety box.

        :return: dictionary containing safety and reachability information
        """
        x = kwargs.get('x')
        y = kwargs.get('y')
        z = kwargs.get('z')
        execute = kwargs.get('execute', True)

        if x is None or y is None or z is None:
            raise ValueError('safe_set_ee_pose_components requires x, y, and z.')

        safe = self.is_pose_safe(x, y, z)
        report = {
            'safe': safe,
            'reachable': False,
            'success': False,
            'executed': bool(execute and safe),
            'status': 'unsafe' if not safe else 'unchecked',
        }
        if not safe:
            return report

        _, reachable = self.arm.set_ee_pose_components(**kwargs)
        report['reachable'] = reachable
        report['success'] = safe and reachable
        report['status'] = 'reached' if reachable else 'unreachable'
        if report['success'] and execute:
            self._record_reached_pose(
                {
                    'x': x,
                    'y': y,
                    'z': z,
                    'roll': kwargs.get('roll'),
                    'pitch': kwargs.get('pitch'),
                    'yaw': kwargs.get('yaw'),
                }
            )
        return report

    def get_lat_long_pose_components(self, lat_index, long_index, target=None):
        """
        Build the end-effector pose for a dome node.

        :param lat_index: dome latitude index
        :param long_index: dome longitude index
        :param target: optional `(x, y, z)` point to point at; defaults to the dome center
        :return: pose dictionary compatible with `set_ee_pose_components(**pose)`
        """
        x, y, z = self.dome.get_lat_long_point(lat_index, long_index)
        if target is None:
            target = (self.dome.x, self.dome.y, self.dome.z)
        xv, yv, zv = target
        return self.dome.get_ee_pose_components_pointing_at(x, y, z, xv, yv, zv)

    def is_lat_long_reachable(
        self,
        lat_index,
        long_index,
        target=None,
        return_report=False,
        **kwargs,
    ):
        """
        Return `True` if the arm can solve the pose for a dome node.

        :param lat_index: dome latitude index
        :param long_index: dome longitude index
        :param target: optional `(x, y, z)` point to point at; defaults to the dome center
        :param kwargs: extra keyword arguments forwarded to `set_ee_pose_components(...)`
        :param return_report: if `True`, return a detailed report dictionary
        :return: `True` if the pose is safe and reachable; otherwise `False`
        """
        command_pose = self.get_lat_long_pose_components(
            lat_index=lat_index,
            long_index=long_index,
            target=target,
        )
        report = self.safe_set_ee_pose_components(
            **command_pose,
            execute=False,
            **kwargs,
        )
        if return_report:
            return report
        return report['safe'] and report['reachable']

    def move_to_lat_long(
        self,
        lat_index,
        long_index,
        target=None,
        return_report=False,
        **kwargs,
    ):
        """
        Move the arm to a dome node.

        :param lat_index: dome latitude index
        :param long_index: dome longitude index
        :param target: optional `(x, y, z)` point to point at; defaults to the dome center
        :param kwargs: extra keyword arguments forwarded to `set_ee_pose_components(...)`
        :param return_report: if `True`, return a detailed report dictionary
        :return: `True` if the move was safe and succeeded; otherwise `False`
        """
        if self._is_current_lat_long(lat_index, long_index):
            report = {
                'safe': True,
                'reachable': True,
                'success': True,
                'executed': False,
                'status': 'already_reached',
            }
            if return_report:
                return report
            return report['success']

        command_pose = self.get_lat_long_pose_components(
            lat_index=lat_index,
            long_index=long_index,
            target=target,
        )
        report = self.safe_set_ee_pose_components(
            **command_pose,
            **kwargs,
        )
        if report['success'] and report['executed']:
            self._record_current_lat_long(lat_index, long_index)
        if return_report:
            return report
        return report['success']

    def go_to_reset_position(self, reset_position='sleep', **kwargs):
        """
        Move the arm to a named reset pose.

        :param reset_position: named reset pose; currently supports `'sleep'` and `'home'`
        :param kwargs: extra keyword arguments forwarded to the arm reset method
        """
        if reset_position == 'sleep':
            self.arm.go_to_sleep_pose(**kwargs)
        elif reset_position == 'home':
            self.arm.go_to_home_pose(**kwargs)
        else:
            raise ValueError(f'Unsupported reset_position: {reset_position}')

    def try_move_to_lat_long(
        self,
        lat_index,
        long_index,
        target=None,
        return_report=False,
        **kwargs,
    ):
        """
        Check reachability and, if possible, move the arm to a dome node.

        :param lat_index: dome latitude index
        :param long_index: dome longitude index
        :param target: optional `(x, y, z)` point to point at; defaults to the dome center
        :param kwargs: extra keyword arguments forwarded to `move_to_lat_long(...)`
        :param return_report: if `True`, include the detailed report dictionary
        :return: tuple of `(success, command_pose, status_string)` by default
        """
        command_pose = self.get_lat_long_pose_components(
            lat_index=lat_index,
            long_index=long_index,
            target=target,
        )
        if self._is_current_lat_long(lat_index, long_index):
            report = {
                'safe': True,
                'reachable': True,
                'success': True,
                'executed': False,
                'status': 'already_reached',
            }
            result = (True, command_pose, 'already_reached')
            if return_report:
                return result + (report,)
            return result

        reachability_report = self.is_lat_long_reachable(
            lat_index=lat_index,
            long_index=long_index,
            target=target,
            return_report=True,
            **kwargs,
        )
        if not reachability_report['safe']:
            result = (False, command_pose, 'unsafe')
            if return_report:
                return result + (reachability_report,)
            return result
        if not reachability_report['reachable']:
            result = (False, command_pose, 'unreachable')
            if return_report:
                return result + (reachability_report,)
            return result

        move_report = self.move_to_lat_long(
            lat_index=lat_index,
            long_index=long_index,
            target=target,
            return_report=True,
            **kwargs,
        )
        if not move_report['success']:
            result = (False, command_pose, move_report['status'])
            if return_report:
                return result + (move_report,)
            return result

        result = (True, command_pose, 'reached')
        if return_report:
            return result + (move_report,)
        return result
