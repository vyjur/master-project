from datetime import datetime, timedelta
from pyvis.network import Network
import plotly
import plotly.express as px
import pandas as pd
from structure.enum import ME, TR_TLINK

class Timeline:
    def __init__(self, config=None, offset=3):
        self.__config = config
        self.__offset = offset

    def create(self, data):
        timeline = []
        
        for doc in data:
            levels = {e.id: e.level for e in doc["entities"]}
            level_dict = {val: 0 for val in levels.values()}
            
            for e in doc["entities"]:
                if e.id not in levels or levels[e.id] is None or e.type is None or e.dct is None:
                    continue
                
                # TODO: change offset
                start_date = e.dct + timedelta(hours=levels[e.id] * self.__offset)
                end_date = e.dct + timedelta(
                    hours=levels[e.id] * self.__offset + self.__offset
                )
                
                # TODO: change level thingy?
                timeline.append(
                    dict(
                        System=level_dict[levels[e.id]],
                        Entity=e.value,
                        Type=e.type.name,
                        Start=start_date,
                        Finish=end_date,
                        Document=doc["dct"],
                    )
                )
                level_dict[levels[e.id]] += 1
        if len(timeline) < 1:
            return
        df = pd.DataFrame(timeline)

        fig = px.timeline(
            df,
            x_start="Start",
            x_end="Finish",
            y="System",
            color="Entity",
            text="Entity",
            custom_data=df[["Type"]]
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
            hovertemplate="Document: %{base|%Y-%m-%d}<br>"
            "Type: %{customdata[0]}",
            textangle=0,
            insidetextfont=dict(size=56),
        )

        #fig.show()
        plotly.offline.plot(fig, filename='./src/visualization/timeline.html')

