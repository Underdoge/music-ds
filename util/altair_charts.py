""" This module includes all the functions used to generate the charts in in
Jupyter Notebook to be used in the streamlit app.
"""
import altair as alt
import polars as pl
from altair import Chart
from polars import DataFrame


def top_10_genres_chart(songs_df: DataFrame) -> Chart:
    """Top 10 genres chart.

    This function returns an altair Chart of the top 10 genres with most
    songs.

    Args:
        songs_df (DataFrame): A polars DataFrame containing the songs with
    their genre.

    Returns:
        Chart: The generated altair chart.
    """
    songs_by_genre = songs_df.select(
        pl.col('Genre')
    ).to_series().value_counts()
    top_10_genres = songs_by_genre.top_k(10, by="counts")
    top_genre = top_10_genres.top_k(1, by="counts").to_numpy()[0][0]
    return alt.Chart(top_10_genres, title="Top 10 Genres").mark_bar().encode(
        x=alt.X('counts', title="Songs"),
        y='Genre',
        color=alt.condition(
            alt.datum.Genre == top_genre,
            alt.ColorValue(alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='darkblue', offset=0),
                       alt.GradientStop(color='purple', offset=.2),
                       alt.GradientStop(color='yellow', offset=1)])),
            alt.ColorValue(alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='darkblue', offset=0),
                       alt.GradientStop(color='purple', offset=1)]))
        )
    )


def top_10_avg_durations_by_genre(songs_df: DataFrame) -> Chart:
    """ Return an altair Chart of the top 10 average song
    durations by genre.

    Args:
        songs_df (DataFrame): A polars DataFrame containing the songs with
    their genre and duration.

    Returns:
        Chart: The generated altair chart.
    """
    avg_duration_per_genre = songs_df.group_by('Genre').agg(
        (pl.mean('Total Time')/60000).round(1).alias('Minutes')
    )
    top_10_avg_duration = avg_duration_per_genre.top_k(10, by="Minutes")
    top_duration = avg_duration_per_genre.top_k(1,
                                                by="Minutes").to_numpy()[0][1]
    return alt.Chart(top_10_avg_duration,
                     title="Top 10 AVG Durations By Genre").mark_bar().encode(
        x=alt.X('Genre', axis=alt.Axis(labelAngle=-45)),
        y="Minutes",
        color=alt.condition(
            alt.datum.Minutes == top_duration,
            alt.ColorValue(alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='darkblue', offset=0),
                       alt.GradientStop(color='purple', offset=.2),
                       alt.GradientStop(color='yellow', offset=1)],
                x1=1,
                x2=1,
                y1=1,
                y2=0)),
            alt.ColorValue(alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='darkblue', offset=0),
                       alt.GradientStop(color='purple', offset=1)],
                x1=1,
                x2=1,
                y1=1,
                y2=0))
            )
    ).properties(width=400)


def songs_per_year(songs_df: DataFrame) -> Chart:
    """ This function returns an altair Chart of all the songs arranged by
    year.

    Args:
        songs_df (DataFrame): A polars DataFrame containing the songs and
    their release year.

    Returns:
        Chart: The generated altair chart.
    """

    songs_per_year = songs_df.select(
        pl.col('Name').alias('Songs'),
        pl.col('Year')
    ).group_by('Year').agg(
        pl.count('Songs')
    ).sort(by='Year')
    top_10_years = songs_per_year.top_k(10, by="Songs")
    top_year = top_10_years.top_k(1, by="Songs").to_numpy()[0][0]
    return alt.Chart(top_10_years,
                     title="Song Count per Year").mark_bar().encode(
                         x=alt.X("Year:N", axis=alt.Axis(labelAngle=-45)),
                         y="Songs",
                         color=alt.condition(
                            alt.datum.Year == top_year,
                            alt.ColorValue(alt.Gradient(
                                gradient='linear',
                                stops=[alt.GradientStop(color='darkblue',
                                                        offset=0),
                                       alt.GradientStop(color='purple',
                                                        offset=.2),
                                       alt.GradientStop(color='yellow',
                                                        offset=1)],
                                x1=1,
                                x2=1,
                                y1=1,
                                y2=0)),
                            alt.ColorValue(alt.Gradient(
                                gradient='linear',
                                stops=[alt.GradientStop(color='darkblue',
                                                        offset=0),
                                       alt.GradientStop(color='purple',
                                                        offset=1)],
                                x1=1,
                                x2=1,
                                y1=1,
                                y2=0))
                         )
    ).properties(width=400)


def songs_by_bitrate(songs_df: DataFrame) -> Chart:
    """ This function returns an altair Chart with the top 5 most used bit
    rates.

    Args:
        songs_df (DataFrame): A polars DataFrame containing the songs with
    their bit rate.

    Returns:
        Chart: The generated altair chart.
    """

    songs_by_bitrate = songs_df.group_by('Bit Rate').agg(
        pl.count('Name').alias('Count')
    ).select(pl.col('Bit Rate').alias('Bitrate'), pl.col('Count'))
    top_5_bitrates = songs_by_bitrate.top_k(5, by="Count")
    return alt.Chart(
        top_5_bitrates,
        title="Top 5 Bit Rates").mark_arc(innerRadius=70).encode(
        color=alt.Color("Bitrate", title="Bit Rate (kb/s)", type="nominal",
                        sort='ascending',
                        scale=alt.Scale(scheme='plasma')),
        theta="Count"
    )


def top_5_genre_count_per_avg_bitrate(songs_df: DataFrame) -> Chart:
    """ This function returns an altair Chart with the number of genres
    using the top 5 average bit rates.

    Args:
        songs_df (DataFrame): A polars DataFrame containing the songs with
    their genre and bit rate.

    Returns:
        Chart: The generated altair chart.
    """

    avg_bitrate_per_genre = songs_df.group_by('Genre').agg(
        pl.mean('Bit Rate').alias('Average Bit Rate')
    )
    top_5_bitrates_per_genre = avg_bitrate_per_genre.group_by(
        'Average Bit Rate').agg(
            pl.count('Genre')
    ).top_k(5, by='Genre').sort(by='Genre')
    return alt.Chart(top_5_bitrates_per_genre,
                     title="Top 5 Genre Count per AVG Bit Rate").encode(
        alt.Theta("Average Bit Rate:N").sort(
            [160.0, 192.0, 96.0, 128.0, 320.0]),
        alt.Radius("Genre").scale(type="sqrt", rangeMin=20),
        color=alt.Color("Average Bit Rate:N",
                        scale=alt.Scale(scheme='plasma'))
    )


def top_10_most_played_songs(scrobbles_df: DataFrame) -> Chart:
    """ This function returns an altair Chart with the top 10 most played
    songs.

    Args:
        scrobbles_df (DataFrame): A polars DataFrame containing a time series
    of every times a song was played.

    Returns:
        Chart: The generated altair chart.
    """

    scrobbles_df = scrobbles_df.with_columns(
        (pl.col('track') + " - " + pl.col('artist')).alias("Song - Artist")
    )
    played_songs = scrobbles_df.group_by('Song - Artist').agg(
        pl.col('Song - Artist').count().alias('play_num')
    ).sort(by="play_num", descending=True)
    top_10_played_songs = played_songs.top_k(10, by="play_num")
    top_song = top_10_played_songs.top_k(1, by="play_num").to_numpy()[0][1]
    return alt.Chart(top_10_played_songs,
                     title="Top 10 Most Played Songs").mark_bar().encode(
                     x=alt.X('play_num', title="Number of times played"),
                     y="Song - Artist",
                     color=alt.condition(
                         alt.datum.play_num == top_song,
                         alt.ColorValue(alt.Gradient(
                            gradient='linear',
                            stops=[alt.GradientStop(color='darkblue',
                                                    offset=0),
                                   alt.GradientStop(color='purple',
                                                    offset=.2),
                                   alt.GradientStop(color='yellow',
                                                    offset=1)])),
                         alt.ColorValue(alt.Gradient(
                            gradient='linear',
                            stops=[alt.GradientStop(color='darkblue',
                                                    offset=0),
                                   alt.GradientStop(color='purple',
                                                    offset=1)]))
                     )
    ).properties(width=600)
