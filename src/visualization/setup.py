from datetime import datetime, timedelta
from pyvis.network import Network
import plotly
import plotly.express as px
import pandas as pd
from structure.enum import ME, TLINK, TIMEX


class Timeline:
    def __init__(self, config=None, offset=24):
        self.__config = config
        self.__offset = offset

    def create(self, data, save_path="./"):

        if not isinstance(data, pd.DataFrame):
            timeline = []

            levels = list(set([e.date for doc in data for e in doc["entities"]]))
            level_dict = {val: 0 for val in levels}

            for doc in data:

                for e in doc["entities"]:
                    if e.type is None or isinstance(e.type, TIMEX) or e.date is None:
                        continue

                    start_date = e.date
                    if not isinstance(start_date, datetime):
                        start_date = self.__normalize_date(
                            start_date
                        )  # Adjust format if needed

                    end_date = start_date + timedelta(hours=self.__offset)

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

            print(df)
            df.to_csv(save_path + "timeline.csv")
        else:
            df = data
        self.__plot(df, save_path)

    def __plot(self, df, save_path):
        fig = px.timeline(
            df,
            x_start="Start",
            x_end="Finish",
            y="System",
            color="Type",
            text="Entity",
            custom_data=df[["Entity"]],
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
        fig.update_yaxes(
            visible=False,
            showticklabels=False,
        )  # Hide the y-axis

        fig.update_layout(barmode="overlay")
        fig.update_traces(
            hovertemplate="Document: %{base|%Y-%m-%d}<br>" "Text: %{customdata[0]}",
            textangle=0,
            insidetextfont=dict(size=56),
        )

        fig.update_layout(
            xaxis=dict(
                tickmode="auto",  # Automatically choose tick spacing
                nticks=5,  # Max number of ticks
                tickformat="%Y-%m-%d",  # Format the ticks to show only the date,
                showgrid=True,
            ),
        )
        
        fig.update_xaxes(
            scaleanchor = "y",  # This makes the x-axis scale dependent on the y-axis
            scaleratio = 5      # This sets the scaling ratio between the x and y axes
        )
        
        df['Start'] = pd.to_datetime(df['Start'])
        x_min = df['Start'].min()
        # Zoom in on a portion (e.g., first 10 units)
        # Zoom in on the x-axis (e.g., show the first 10 days)
        zoom_factor = pd.Timedelta(days=30)  # 10 days zoom
        fig.update_xaxes(range=[x_min, x_min + zoom_factor])



        # This is if we want to get the plot up right away
        # fig.show()
        plotly.offline.plot(fig, filename=save_path + "timeline.html")

    def __normalize_date(self, date_str):
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                dt = datetime.strptime(date_str, fmt)
                if fmt == "%Y":
                    return dt.replace(
                        month=1, day=1
                    )  # Return datetime for the first day of the year
                elif fmt == "%Y-%m":
                    return dt.replace(
                        day=1
                    )  # Return datetime for the first day of the month
                else:
                    return dt  # Return the exact datetime object
            except ValueError:
                continue
        raise ValueError(f"Unsupported date format: {date_str}")


if __name__ == "__main__":
    import random

    class MockEntity:
        def __init__(self, value, date, type_):
            self.value = value
            self.date = date
            self.type = type_

    def generate_sparse_dates(start_year=2024, end_year=2025, num_dates=15):
        """Generate a list of sparse datetime objects between two years."""
        sparse_dates = []
        for _ in range(num_dates):
            year = random.randint(start_year, end_year)
            month = random.randint(1, 12)
            day = random.randint(1, 28)  # Keep it simple and valid
            sparse_dates.append(datetime(year, month, day))
        return sorted(sparse_dates)

    def mock_sparse_data():
        sparse_dates = generate_sparse_dates()
        docs = []

        for i, date in enumerate(sparse_dates):
            # One entity per doc for simplicity
            ent = MockEntity(
                f"Event {i}", date, ME.CONDITION if i % 3 != 0 else ME.TREATMENT
            )
            docs.append({"dct": f"Doc_{i}", "entities": [ent]})

        return docs

    timeline = Timeline(offset=24)  # Wider offset for visibility
    # timeline.create(mock_sparse_data(), save_path="./")

    df = pd.read_csv(
        "data/helsearkiv/evaluate/tem1/data/helsearkiv/safe-patients/3/timeline.csv"
    )
    timeline.create(df, save_path="./app/templates/output/")
