"""
This is a streamlit app that will showcase an hypothetical use case for the
model created under 'models/music_size.joblib'.
"""
import streamlit as st
import polars as pl
from joblib import load
from util.on_change import update_slider, update_numin
from util.scaler import scale
from util.data_imputation import data_imputation
import plotly.express as px
import numpy as np

# "st.session_state object:", st.session_state

st.markdown(
    """
# Music storage size prediction :notes:
## Hypothetical case study
### Estimating the required storage for a music \
streaming company.

A music streaming company approached us to help them estimate how much space \
they would require to store their music files if they offered them to their \
users in different bit rates.

They provided us their current music files, and we were able to create a model\
 to predict the required total storage size if all the songs were converted to\
 the same bit rate.
"""
)

# Import dataset

original_df = pl.read_csv('data/music_library_export.csv')

# Data imputation

songs_df = data_imputation(original_df)

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

# Add results to new_df

size_series = pl.Series("Size", np.concatenate(predicted_size))

new_df = new_df.with_columns(
    size_series
).rename({"Total Time": "Duration"})

# Remove negative sizes

new_df = new_df.select(
    pl.col('Duration').filter(pl.col('Size') > 0),
    pl.col('Bit Rate').filter(pl.col('Size') > 0),
    pl.col('Size').filter(pl.col('Size') > 0)
)

# Plot results

fig_2 = px.scatter_3d(
    new_df,
    x='Duration',
    y='Bit Rate',
    z='Size',
    color='Size'
)

st.plotly_chart(fig_2, theme=None)
