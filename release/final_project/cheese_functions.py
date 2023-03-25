#######################
### Import Packages ###
#######################

import pandas as pd
import altair as alt
from IPython.display import display

####################
### Score Models ###
####################


def get_scores_chart(scores_df, scoring, fig_number, baseline_model="Dummy Classifier"):
    """
    Returns a histogram chart of a dataframe feature

    Parameters
    ----------
    scores_df: pandas.core.frame.DataFrame
        the dataframe with models and their scoring data
    scoring: str, optional
        the baseline model
    scoring: str
        the column to score on
    fig_number: int
        the figure number to display in the chart title
    baseline_model :
        the baseline model for comparing test scores and metrics

    Returns
    -------
    next_number : int
        the next figure number to display in the chart_title

    Displays
    -------
    altair.vegalite.v3.api.Chart
        an Altair histogram

    Examples
    --------
    >>> get_scores_chart(
            scores_df = scores_df,
            scoring = 'precision',
            fig_number = fig_number
        )
    """

    if type(scores_df) != pd.DataFrame:
        raise TypeError("scores_df must be a pandas.core.frame.DataFrame object.")
    if "model" not in scores_df.columns.to_list():
        raise ValueError("scores_df must contain a column for the models being scored.")
    if len(scores_df.where(scores_df["model"] == baseline_model).dropna()) != 1:
        raise ValueError(f"""{baseline_model} must be found in the 'model' column.""")

    # Check if Scoring String is Found Within DataFrame Columns.
    found = False
    for col in scores_df.columns.to_list():
        if col.find(scoring) != -1:
            found = True
    if not found:
        raise ValueError(
            "The scoring_col must be a substring within the dataframe columns."
        )

    # Melt DataFrame for Plotting Scores.
    plot_df = pd.melt(
        frame=scores_df,
        id_vars="model",
        var_name="score_type",
        value_name="score",
        value_vars=scores_df.columns.to_list().remove("model"),
    )

    # Assign the DataFrame to the Effective Score Type.
    plot_df = (
        plot_df[plot_df["score_type"].str.find(scoring) != -1]
        .reset_index(drop=True)
        .sort_values(by="score_type", ascending=True)
    )
    display(plot_df.head())

    # Assign Labels.
    score_label = " ".join([word.capitalize() for word in scoring.split("_")])
    x_title = "Score" if scoring.find("time") == -1 else "Time (s)"

    y_order = [baseline_model] + sorted(
        [model for model in list(plot_df["model"].unique()) if model != baseline_model]
    )

    # Create Altair Chart.
    scores_chart = (
        alt.Chart(
            plot_df,
            title=alt.TitleParams(
                f"Figure {fig_number} : Model Performance - {score_label} Score",
                fontSize=20,
            ),
        )
        .mark_bar(opacity=0.75)
        .encode(
            x=alt.X(
                "score:Q",
                title=x_title,
                scale=alt.Scale(
                    domain=[
                        plot_df.score.min() - 0.025,
                        min(1.0, plot_df.score.max() + 0.025),
                    ]
                ),
                stack=False,
            ),
            y=alt.Y("model:N", title="Model", sort=y_order),
            color=alt.Color(
                "score_type:N",
                legend=alt.Legend(
                    title="Score Type", titleFontSize=14, labelFontSize=12
                ),
            ),
            tooltip=[alt.Tooltip("score:Q", title="Score")],
        )
        .properties(height=200, width=500)
        .configure_axis(labelFontSize=15, titleFontSize=20)
    )

    # Display Scores Chart
    display(scores_chart)

    next_number = fig_number + 1

    return next_number


#########################################
### Feature Value Distribution Charts ###
#########################################


def get_numeric_chart(plot_df, feature, fig_number, num_bins=25):
    """
    Returns a histogram chart of a dataframe feature

    Parameters
    ----------
    plot_df: pandas.core.frame.DataFrame
        the dataframe to plot
    feature: str
        the feature name
    fig_number: int
        the figure number to display in the chart title
    maxbins: int, optional
        the maximum number of data bins on the y-axis

    Returns
    -------
    altair.vegalite.v3.api.Chart
        an Altair histogram

    Examples
    --------
    >>> get_numeric_chart(
            plot_df = X_train,
            feature = 'MoisturePercent',
            fig_number = 4
        )
    """

    if type(plot_df) != pd.DataFrame:
        raise TypeError("plot_df must be a pandas.core.frame.DataFrame object.")
    if feature not in plot_df.columns.to_list():
        raise ValueError("The feature must be within the dataframe columns.")
    if feature not in plot_df.describe(include=["int64", "float64"]).T.index.to_list():
        raise TypeError("The column must include a numeric feature.")

    plot_df = plot_df.copy()

    # Check if Binary Feature.
    if len(plot_df[feature].unique()) != 2:
        bin = alt.Bin(maxbins=num_bins)
        x_shorthand = f"{feature}:Q"
    else:
        plot_df[feature] = plot_df[feature].map({0: False, 1: True})

        bin = False
        x_shorthand = f"{feature}:N"

    # Create Altair Chart.
    numeric_chart = (
        alt.Chart(
            plot_df,
            title=alt.TitleParams(
                f"Figure {fig_number} : {feature} Feature Value Distributions",
                fontSize=20,
            ),
        )
        .mark_bar()
        .encode(
            x=alt.X(x_shorthand, bin=bin, title=f"{feature} Values"),
            y=alt.Y("count():Q", title="Number of Records"),
            tooltip=[alt.Tooltip("count():Q", title="Number of Records")],
        )
        .properties(height=300, width=400)
        .configure_axis(labelFontSize=12.5, titleFontSize=15)
    )

    return numeric_chart


def get_categorical_chart(plot_df, feature, fig_number, sort_order=None):
    """
    Returns a bar chart of a dataframe feature

    Parameters
    ----------
    plot_df: pandas.core.frame.DataFrame
        the dataframe to plot
    feature: str
        the feature name
    fig_number: int
        the figure number to display in the chart title
    sort_order : list, optional
        the order to sort the feature values in the chart

    Returns
    -------
    altair.vegalite.v3.api.Chart
        an Altair histogram

    Examples
    --------
    >>> get_categorical_chart(
            plot_df = X_train,
            feature = 'ManufacturingTypeEn',
            fig_number = 5
        )
    """

    if type(plot_df) != pd.DataFrame:
        raise TypeError("plot_df must be a pandas.core.frame.DataFrame object.")
    if feature not in plot_df.columns.to_list():
        raise ValueError("The feature must be within the dataframe columns.")
    if feature not in plot_df.describe(include="object").T.index.to_list():
        raise TypeError("The column must include a categorical feature.")

    sort = sort_order if sort_order else "x"

    # Create Altair Chart.
    categorical_chart = (
        alt.Chart(
            plot_df,
            title=alt.TitleParams(
                f"Figure {fig_number} : {feature} Feature Value Distributions",
                fontSize=20,
            ),
        )
        .mark_bar()
        .encode(
            alt.X("count():Q", title="Number of Records"),
            alt.Y(f"{feature}:N", sort=sort),
            tooltip=[alt.Tooltip("count():Q", title="Number of Records")],
        )
        .properties(height=300, width=250)
        .configure_axis(labelFontSize=12.5, titleFontSize=15)
    )

    return categorical_chart


################################
### Describe Dataset Feature ###
################################


def describe_features(effective_df, features, fig_number, sort_by=None):
    """
    Prints distinct values for each feature in a list of dataframe features

    Parameters
    ----------
    effective_df: pandas.core.frame.DataFrame
        the dataframe to plot
    features: list
        list of feature names
    fig_number: int
        the figure number to display in the chart title
    sort_order : list, optional
        the order to sort the feature values in the chart

    Returns
    -------
    next_number : int
        the next figure number to display in the chart_title

    Displays
    -------
    altair.vegalite.v3.api.Chart
        an Altair histogram

    """

    if type(effective_df) != pd.DataFrame:
        raise TypeError("effective_df must be a pandas.core.frame.DataFrame object.")

    for feature in features:
        print(
            f"""The distinct values in the {feature} column are : \n{list(effective_df[feature].unique())}"""
        )

        if (
            feature
            in effective_df.describe(include=["int64", "float64"]).T.index.to_list()
        ):
            plot_df = effective_df.copy().fillna(0)
            feature_chart = get_numeric_chart(
                plot_df=plot_df,
                feature=feature,
                fig_number=fig_number + features.index(feature),
            )

        elif feature in effective_df.describe(include="object").T.index.to_list():
            plot_df = effective_df.copy().fillna("nan")
            sort_order = (
                sort_by
                if (sort_by in ["x", "-x"])
                else sorted(list(plot_df[feature].unique()))
            )

            feature_chart = get_categorical_chart(
                plot_df=plot_df,
                feature=feature,
                fig_number=fig_number + features.index(feature),
                sort_order=sort_order,
            )
        else:
            raise ValueError("The feature must be within the dataframe columns.")

        # Display Feature Chart
        display(feature_chart)

    next_number = fig_number + len(features)

    return next_number
