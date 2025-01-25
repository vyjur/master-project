from pyvis.network import Network
from structure.enum import ME, TR_TLINK


class VizTool:
    def __init__(self, config=None):
        self.config = config
        # TODO: add config
        self.net = Network(
            notebook=True,
            height="500px",
            width="100%",
            bgcolor="#222222",
            font_color="white",  # type: ignore
            directed=True,
            neighborhood_highlight=True,
            filter_menu=True,
            layout=True,
        )

    def clear(self):
        # TODO: fix this
        if self.config == None:
            self.net = Network()
        else:
            self.net = Network(*self.config)  # type: ignore

    def create(self, entities):
        self.clear()
        for entity in entities:
            match entity.type:
                case ME.CONDITION:
                    color = "#F05D5E"
                case ME.EVENT:
                    color = "#8390FA"
                case ME.SYMPTOM:
                    color = "#FAC748"
                case _:
                    color = "grey"
                   
            if entity.type is None:
                continue
            self.net.add_node(
                entity.id, entity.value, color=color, title=entity.type.name
            )

        for entity in entities:
            for rel in entity.relations:
                if rel.tr != TR_TLINK.XDURINGY:
                    self.net.add_edge(
                        entity.id,
                        rel.y.id,
                        title=rel.er.name if rel.er is not None else "",
                    )
                else:
                    # TODO: fix here
                    if rel.er == ER.EQUAL:
                        color = "grey"
                        self.net.add_edge(
                            entity.id,
                            rel.y.id,
                            color=color,
                            title=rel.er.name if rel.er is not None else "",
                        )

        # self.net.show_buttons(filter_=['renderer', 'layout'])
        # Enable physics
        self.net.toggle_physics(True)

        # Show the graph and embed it in the notebook
        html_file = "output.html"
        self.net.show(html_file, notebook=False)

