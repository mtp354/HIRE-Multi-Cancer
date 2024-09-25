import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import subprocess
from io import StringIO

# Load data and compute static values
from shared import app_dir, src_dir
import src.main as main
from shinywidgets import output_widget, render_plotly
from shinyswatch import theme

from shiny import App, reactive, render, ui

# Add page title and sidebar
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_selectize(
            id="site",
            label="Cancer Site",
            choices=["Lung", "Colorectal", "Pancreas", "Breast", "Prostate"],
            multiple=True,
            options=(
            {
                "placeholder": "Select site(s)",
            }
        )
        ),
        ui.input_radio_buttons(
            id="race",
            label="Race",
            choices=["NH Black", "NH White"],
            selected="NH Black",
            inline=True,
        ),
        ui.input_radio_buttons(
            id="sex",
            label="Sex",
            choices=["Female", "Male"],
            selected="Female",
            inline=True,
        ),
        ui.input_slider(
            "cohort",
            "Birth Cohort",
            min=1930,
            max=1960,
            step=1,
            sep="",
            value=1930,
            drag_range=False,
        ),
        ui.input_slider(
            "age_interval",
            "Age Interval",
            min=0,
            max=100,
            step=1,
            sep="",
            value=(0, 100),
        ),
        ui.input_action_button("plot_incidence_btn", "Plot Incidence"),
        open="desktop",
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header(
                "Incidence",
                class_="d-flex justify-content-between align-items-center",
            ),
            # ui.output_ui("selected_params"),
            ui.h6(
                "Select parameters and click Plot Incidence to view plot",
                id="intro_text",
                class_="card-text d-flex justify-content-center align-items-center flex-grow-1",
            ),
            output_widget("plot_incidence"),
            full_screen=True,
            id="plot_card",
        ),
    ),
    ui.include_css(app_dir / "styles.css"),
    title="Multi-Cancer Model",
    fillable=True,
    theme=theme.zephyr(),
)


def server(input, output, session):
    show_initial = reactive.value[bool](True)

    # @render.ui
    # def selected_params():
    #     return f"Site: {input.site()}, Race: {input.race()}, Sex: {input.sex()}, Birth Cohort: {input.cohort()}, Ages: {input.age_interval()[0]}-{input.age_interval()[1]}"

    @reactive.effect
    @reactive.event(input.plot_incidence_btn)
    def remove_intro_text():
        # Hide intro text on button click
        show_initial.set(False)
        ui.remove_ui("#intro_text")

    @render_plotly
    # @reactive.event(input.plot_incidence_btn)
    def plot_incidence():
        input.plot_incidence_btn()

        with reactive.isolate():
            if show_initial():
                return go.Figure(
                    go.Scatter(
                        x=pd.Series(dtype=object),
                        y=pd.Series(dtype=object),
                        mode="markers",
                    )
                )

            inc_df = pd.DataFrame()

            # # Call main.py with the specified arguments
            # result = subprocess.run(
            #     [
            #         "python3",
            #         str(src_dir / "main.py"),
            #         "--mode",
            #         "app",
            #         "--sojourn_time",
            #         "False",
            #         "--cohort_year",
            #         str(input.cohort()),
            #         "--start_age",
            #         str(input.age_interval()[0]),
            #         "--end_age",
            #         str(input.age_interval()[1]),
            #         "--cohort_sex",
            #         input.sex(),
            #         "--cohort_race",
            #         input.race()[3:],
            #         "--cancer_sites",
            #         str(input.site()),
            #         "--cancer_sites_ed",
            #         "",
            #     ],
            #     env={"PYTHONPATH": str(src_dir.parent)},
            #     capture_output=True,
            #     text=True,
            #     check=False,
            # )

            # if result.returncode != 0:  # Error
            #     print("Error:", result.stderr)
            #     ui.insert_ui(
            #         ui.h6(
            #             f"Error: {result.stderr}",
            #             id="plot_error_text",
            #             class_="card-text d-flex justify-content-center align-items-center flex-grow-1",
            #             style="color: red;",
            #         ),
            #         selector="#plot_card",
            #     )
            #     return go.Figure(
            #         go.Scatter(
            #             x=pd.Series(dtype=object),
            #             y=pd.Series(dtype=object),
            #             mode="markers",
            #         )
            #     )
            
            # Remove text from a previous error
            ui.remove_ui("#plot_error_text")

            # inc_df = pd.read_csv(StringIO(result.stdout))
            inc_df = main.run_model(
                cancer_sites=input.site(),
                cohort=input.cohort(),
                sex=input.sex(),
                race=input.race(),
                start_age=input.age_interval()[0],
                end_age=input.age_interval()[1],
                save=False,
            )
            inc_df = inc_df.iloc[:-1]
            fig = px.line(
                data_frame=inc_df,
                x="Age",
                y="Incidence per 100k",
                title=f"{input.race()} {input.sex()} {str(input.cohort())}",
            )
            fig.update_layout(title_x=0.5)
            return fig

    @reactive.effect
    @reactive.event(input.site)
    def update_sex_options():
        """Remove other sex option when a single sex cancer is selected."""
        female_only = ["Breast"]
        male_only = ["Prostate"]
        site = input.site()
        choices = ["Female", "Male"]
        updated_sex_selected = input.sex()
        if all(s in female_only for s in site):
            updated_sex_selected = "Female"
            choices = [updated_sex_selected]
        elif all(s in male_only for s in site):
            updated_sex_selected = "Male"
            choices = [updated_sex_selected]

        ui.update_radio_buttons(
            id="sex",
            label="Sex",
            choices=choices,
            selected=updated_sex_selected,
            inline=True,
        )


app = App(app_ui, server)
