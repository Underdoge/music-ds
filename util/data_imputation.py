""" This module contains the 'data_imputation' function that cleans up the
provided dataframe.
"""
import polars as pl
from polars import DataFrame


def data_imputation(songs_df: DataFrame) -> DataFrame:
    """This function will clean up the provided DataFrame and return it.

    Args:
        songs_df (DataFrame): A dataframe with the songs dataset.

    Returns:
        DataFrame: The cleaned up polars DataFrame.
    """
    genres = songs_df.select(
        pl.col("Genre").map_elements(
            lambda x: "Alternative" if x == "Alternativo" else x)
        .map_elements(
            lambda x: "Electronic" if x == "Electronica" else x)
        .map_elements(
            lambda x: "Electronic Pop" if x in ["Pop Electronica",
                                                "Electronica / Pop"] else x)
        .map_elements(
            lambda x: "Indie" if x == "indie" else x)
        .map_elements(
            lambda x: "Indie Rock" if x in ["Rock/Indie", "Indie/Rock",
                                            "General Indie Rock"] else x)
        .map_elements(
            lambda x: "Miscellaneous" if x == "misc" else x)
        .map_elements(
            lambda x: "Soundtrack" if x in ["soundtrack",
                                            "Banda sonora"] else x)
        .map_elements(
            lambda x: "Thrash Metal" if x == "Thrash Metal" else x)
        .map_elements(
            lambda x: "Alt Rock" if x in ["Alt. Rock", "Alternative Rock",
                                          "Rock alternativo",
                                          "Alternative, Rock",
                                          "General Alternative Rock"] else x)
        .map_elements(
            lambda x: "Brit Pop" if x == "Brit-pop" else x)
        .map_elements(
            lambda x: "Pop Rock" if x in ["Pop/Rock",
                                          "Pop/Rock 2000's"] else x)
        .map_elements(
            lambda x: "Pop" if x == "General Pop" else x)
        .map_elements(
            lambda x: "Folk" if x == "General Folk" else x)
        .map_elements(
            lambda x: "Rock" if x in ["General Rock", "Rock En General",
                                      "Rock en general", "Rock @",
                                      "rock"] else x)
        .map_elements(
            lambda x: "Heavy Metal" if x == "Rock Duro Y Heavy" else x)
        .map_elements(
            lambda x: "Hip Hop/Rap" if x == "General Rap/Hip-Hop" else x)
        .map_elements(
            lambda x: "Bitpop" if x == "bitpop" else x)
        .map_elements(
            lambda x: "Chillstep" if x == "chillstep" else x)
        .map_elements(
            lambda x: "Chiptune" if x == "chiptune" else x)
        .map_elements(
            lambda x: None
            if x in ["genre", "default", ".", "(255)", "Other"] else x)
        .map_elements(
            lambda x: "Unclassifiable" if x == "General Unclassifiable" else x)
        .map_elements(
            lambda x: "Soft Rock / Alternative Folk / Folk / Rock"
            if x == "soft rock/alternative folk/folk/rock" else x)
        .alias("Genre")
    ).to_series()

    songs_df = songs_df.with_columns(genres.alias("Genre"))
    songs_df = songs_df.drop_nulls(["Genre"])
    songs_df = songs_df.drop_nulls(["Year"])
    songs_df = songs_df.filter(pl.col('Year') > 1000).filter(
        pl.col('Year') < 2024)
    songs_df.drop_in_place("Comments")
    songs_df.drop_in_place("File Folder Count")
    songs_df.drop_in_place("Library Folder Count")
    songs_df.drop_in_place("Kind")
    songs_df.drop_in_place("Location")

    return songs_df
