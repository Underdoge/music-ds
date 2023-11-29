""" This module defines the 'scale' function to transform the 'Total Time'
column of a Polars DataFrame, leaving the 'Bit Rate' column intact.
"""

from numpy import ndarray
from polars import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import RobustScaler


def scale(df: DataFrame) -> ndarray:
    """ This function takes a Polars Dataframe with a "Total Time" and
    "Bit Rate" columns and transforms the "Total Time" one using a RobutScaler.

    Since scikit-learn doesn't support Polars DataFrames direclty because of
    their lack of indexes/iloc methods, we convert the Polars DataFrame
    into a Pandas Dataframe on the fly using its .to_pandas() method.

    Args:
        df (DataFrame): A Polars DataFrame with 'Total Time' and 'Bit Rate'
        columns.

    Returns:
        ndarray (numpy.ndarray): A numpy ndarray with the transformed features.

    """

    scaler = ColumnTransformer([
        ("scaler", RobustScaler(), ["Total Time"])
    ])
    passthrough = ColumnTransformer([
        ("passthrough", "passthrough", ["Bit Rate"])
    ])
    feature_engineering_pipeline = Pipeline([
        ("features", FeatureUnion([
            ("scaled", scaler),
            ("pass", passthrough),]),)
    ])
    return feature_engineering_pipeline.fit_transform(df.to_pandas())
