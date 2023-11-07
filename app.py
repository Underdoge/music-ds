import streamlit as st
import polars as pl
from joblib import load
from util.on_change import update_slider, update_numin
from util.scaler import scale
import plotly.express as px
import numpy as np

# "st.session_state object:", st.session_state

st.markdown(
    """
# Music storage size prediction :notes:
## Hypothetical case study
### Estimating the required storage for a music \
streaming company.

A music company approached us to help them estimate how much space they would \
require to store their music files if they offered them to their users in \
different bit rates.

They provided us their current music files, and we were able to create a model\
 to predict the required total storage size if all the songs were converted to\
 the same bit rate.
"""
)

songs_df = pl.read_csv('data/music_library_export.csv')
songs_size = songs_df.select(pl.col('Size'))
songs_independent_vars = songs_df.select((pl.col('Total Time')/1000),
                                         pl.col('Bit Rate'))

# Data imputation

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
        lambda x: "Soundtrack" if x in ["soundtrack", "Banda sonora"] else x)
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
        lambda x: "Pop Rock" if x in ["Pop/Rock", "Pop/Rock 2000's"] else x)
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
songs_df.select(
    pl.col('Genre')
).unique().sort("Genre")

songs_df = songs_df.drop_nulls(["Genre"])
songs_df = songs_df.drop_nulls(["Year"])
songs_df = songs_df.filter(pl.col('Year') > 1000).filter(pl.col('Year') < 2024)
songs_df.drop_in_place("Comments")
songs_df.drop_in_place("File Folder Count")
songs_df.drop_in_place("Library Folder Count")
songs_df.drop_in_place("Kind")
songs_df.drop_in_place("Location")

# End of Data Imputation

st.markdown(
    """
### Original Data

Here we can see their original songs duration, bit rate and \
size, and their total storage size requirement.
""")

# Display main original data's columns

original_data = songs_df.select(
    pl.col('Artist'),
    pl.col('Name'),
    (pl.col('Total Time')/1000).alias('Duration (seconds)'),
    pl.col('Bit Rate').alias('Bit Rate (kb/s)')).with_columns(
                           songs_df.select(pl.col('Size').alias(
                               'Size (bytes)'))
)
st.dataframe(original_data, hide_index=True)

# Display original dataset's total size

total_original_size = songs_df.select(
    pl.col('Size').sum())
st.write('Total storage size (GBs):',
         round(total_original_size.to_numpy()[0][0]/(1024*1024*1024), 2))

st.markdown(
    """
### Visualization
Now let's see an interactive 3D chart of the 'Size, 'Total Time' and
'Bit Rate' columns.
""")

data = songs_df.select(pl.col('Bit Rate'), pl.col('Size')).with_columns(
    songs_df.select((pl.col('Total Time')/1000).alias('Duration')))

# Plot original data

fig = px.scatter_3d(
    data,
    x='Duration',
    y='Bit Rate',
    z='Size',
    color='Size'
)
st.plotly_chart(fig, theme=None)

st.markdown(
    """
By manipulating the previous 3D chart, we can see that the size of a song most
of the times increases proportionally to its duration and its bit rate. The
song with the biggest size (represented by the yellow dot) also has the
longest duration of all songs, and the blue dots have the smallest size
because they also have either short durations and/or low bit rates.

This gives us a hint we can predict the size of a song by fitting a linear
regression using its duration and bit rate.
""")

st.sidebar.title("Music storage size prediction :notes:")

# Create sidebar input and sidebar linked to each other with update functions

bit_rate = st.sidebar.number_input(
    "Bit Rate (kb/s)", step=64, key='numeric', min_value=128, max_value=2816,
    on_change=update_slider)

slider_val = st.sidebar.slider(
    "Bit Rate (kb/s)", min_value=128, max_value=2816, step=64,
    label_visibility='hidden',
    key='slider',
    on_change=update_numin)

st.markdown(
    """
### Storage Size Prediction
Now we can play with the model and see how the total predicted storage will \
update automatically depending on the selected bit rate by using the "Bit \
Rate (kb/s)" slider or the input field on the left sidebar, so the company \
can make the best decision according to their budget.
"""
)

# Loading the model we created before

model = load('models/music_size.joblib')

# Separate variables

songs_independent_vars = songs_df.select((pl.col('Total Time')/1000),
                                         pl.col('Bit Rate'))
st.write("New bit rate (kb/s):", bit_rate)
new_df = songs_independent_vars.with_columns(
    pl.col('Bit Rate').map_elements(lambda x: bit_rate)
)

# Scale dataset

scaled_new_df = scale(new_df)

# Predict new sizes

predicted_size = model.predict(scaled_new_df)

# Display predicted total size

st.write("Total predicted storage size (GBs): ",
         round(predicted_size.sum()/(1024*1024*1024), 2))

# Fix negative sizes

size_series = pl.Series("Size", np.concatenate(predicted_size)).map_elements(
    lambda x: -x if x < 0 else x
)

# Plot results

new_df = new_df.with_columns(
    size_series
).rename({"Total Time": "Duration"})

fig_2 = px.scatter_3d(
    new_df,
    x='Duration',
    y='Bit Rate',
    z='Size',
    color='Size'
)

st.plotly_chart(fig_2, theme=None)
