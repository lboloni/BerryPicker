"""Create complete robot controllers from experiment/run configurations."""

from robot_controller.graph_robot_controller import GraphRobotController


def create_controller(exp, **kwargs):
    """Construct the graph-level controller selected by ``exp['class']``."""
    controller_class = exp.get("class")
    if controller_class == "GraphRobotController":
        return GraphRobotController(exp, **kwargs)
    raise ValueError(f"Unknown robot controller class {controller_class!r}")
