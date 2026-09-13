"""Graphviz visualization for resolved robot-controller component graphs."""

from html import escape

from graphviz import Digraph


class RCCOVisualizer:
    """Build a Graphviz graph from ``load_controller_spec`` output."""

    DEFAULT_COLORS = {
        "Input": "lightgray",
        "Output": "lightgray",
        "SP_VAE": "lightskyblue",
        "SP_CNN": "lightskyblue",
        "LSTM": "moccasin",
        "MDN": "moccasin",
        "MLP": "moccasin",
    }

    def __init__(
        self, spec, colors=None, *, component_colors=None,
        component_annotations=None, component_styles=None,
    ):
        self.spec = spec
        self.colors = dict(self.DEFAULT_COLORS)
        if colors is not None:
            self.colors.update(colors)
        self.component_colors = component_colors or {}
        self.component_annotations = component_annotations or {}
        self.component_styles = component_styles or {}
        self._validate_spec()

    def _validate_spec(self):
        required = {"name", "components", "connections", "topological_order"}
        if not isinstance(self.spec, dict) or not required.issubset(self.spec):
            raise ValueError(
                "Controller spec must contain name, components, connections, "
                "and topological_order"
            )
        if set(self.spec["topological_order"]) != set(self.spec["components"]):
            raise ValueError(
                "Controller spec topological_order must contain every component"
            )

    @staticmethod
    def _format_ports(ports):
        values = (
            name if size is None else f"{name}[{size}]"
            for name, size in ports.items()
        )
        return "<BR/>".join(escape(value) for value in values) or "-"

    def _node_label(self, label, component):
        input_text = self._format_ports(component["inputs"])
        output_text = self._format_ports(component["outputs"])
        fill_color = self.component_colors.get(
            label, self.colors.get(component["type"], "white")
        )
        annotation = self.component_annotations.get(label)
        annotation_row = ""
        if annotation:
            annotation_row = (
                f'<TR><TD COLSPAN="2"><FONT POINT-SIZE="9">'
                f'{escape(annotation)}</FONT></TD></TR>'
            )
        return (
            f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" '
            f'CELLPADDING="6" BGCOLOR="{escape(fill_color)}">'
            f'<TR><TD COLSPAN="2"><B>{escape(label)}</B></TD></TR>'
            f'<TR><TD COLSPAN="2"><FONT POINT-SIZE="10">'
            f'{escape(component["type"])}</FONT></TD></TR>'
            f'<TR><TD PORT="inputs"><B>Inputs</B><BR/>{input_text}</TD>'
            f'<TD PORT="outputs"><B>Outputs</B><BR/>{output_text}</TD></TR>'
            f'{annotation_row}'
            "</TABLE>>"
        )

    def build(self):
        """Return the configured ``graphviz.Digraph`` without rendering it."""
        dot = Digraph(comment=self.spec["name"])
        dot.attr(rankdir="LR", nodesep="0.5", ranksep="1.0")

        for label in self.spec["topological_order"]:
            component = self.spec["components"][label]
            dot.node(
                label,
                self._node_label(label, component),
                shape="plain",
                style=self.component_styles.get(label, "solid"),
            )

        for connection in self.spec["connections"]:
            edge_text = (
                f"{connection['from_output']} → {connection['to_input']}"
            )
            dot.edge(
                f"{connection['from_component']}:outputs",
                f"{connection['to_component']}:inputs",
                label=edge_text,
                fontsize="10",
            )
        return dot
