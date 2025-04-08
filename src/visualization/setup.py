from datetime import datetime, timedelta
from pyvis.network import Network
import plotly
import plotly.express as px
import pandas as pd
from structure.enum import ME, TLINK, TIMEX

class Timeline:
    def __init__(self, config=None, offset=3):
        self.__config = config
        self.__offset = offset

    def create(self, data, save_path="./"):
        timeline = []
        
        levels = list(set([e.date for doc in data for e in doc["entities"]]))
        level_dict = {val: 0 for val in levels}
        
        for doc in data:
            
            for e in doc["entities"]:
                if e.type is None or isinstance(e.type, TIMEX) or e.date is None:
                    continue
                
                start_date = e.date
                end_date = start_date + timedelta(
                    hours=self.__offset
                )
                
                timeline.append(
                    dict(
                        System=level_dict[e.date],
                        Entity=e.value,
                        Type=e.type.name,
                        Start=start_date,
                        Finish=end_date,
                        Document=doc["dct"],
                    )
                )
                level_dict[e.date] += 1
        
        print("Timeline length:", len(timeline))
        if len(timeline) < 1:
            print("Empty timeline")
            return
        df = pd.DataFrame(timeline)

        fig = px.timeline(
            df,
            x_start="Start",
            x_end="Finish",
            y="System",
            color="Type",
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

        # This is if we want to get the plot up right away
        #fig.show()
        plotly.offline.plot(fig, filename=save_path + 'timeline.html')

