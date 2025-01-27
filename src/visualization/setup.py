from datetime import datetime, timedelta
from pyvis.network import Network
import plotly.express as px
import pandas as pd
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


class Timeline:
    def __init__(self, config=None, offset=3):
        self.__config = config
        self.__offset = offset

    def create(self, data):
        timeline = []
        
        print(data)

        for doc in data:
            levels = {e.id: e.level for e in doc["entities"]}
            print("AHHA", levels)

            level_dict = {val: 0 for val in levels.values()}
            doc["dct"] = datetime(2025, 1, 25, 00, 00, 00)  # Example datetime
            for e in doc["entities"]:
                start_date = doc["dct"] + timedelta(hours=levels[e.id] * self.__offset)
                end_date = doc["dct"] + timedelta(
                    hours=levels[e.id] * self.__offset + self.__offset
                )
                timeline.append(
                    dict(
                        System=level_dict[levels[e.id]],
                        Entity=e.value,
                        Start=start_date,
                        Finish=end_date,
                        Document=doc["dct"],
                    )
                )
                level_dict[levels[e.id]] += 1

        df = pd.DataFrame(timeline)

        fig = px.timeline(
            df,
            x_start="Start",
            x_end="Finish",
            y="System",
            color="Entity",
            text="Entity",
        )
        fig.update_yaxes(
            autorange="reversed"
        )  # otherwise tasks are listed from the bottom up
        fig.update_traces(textposition="inside")
        fig.update_layout(
            xaxis=dict(
                tickformat="%Y-%m-%d",  # Format the ticks to show only the date,
                dtick="D1",
            )
        )
        fig.update_yaxes(visible=False, showticklabels=False)  # Hide the y-axis

        fig.update_layout(barmode="overlay")
        fig.update_traces(
            hovertemplate="Document %{base|%Y-%m-%d}<br>",
            textangle=0,
            insidetextfont=dict(size=56),
        )

        fig.show()

