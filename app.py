"""Streamlit app code.

This Streamlit app will showcase an hypothetical use case for the
'music_size.joblib' and 'genre_prediction.joblib' models under the 'models'
folder.
"""
import numpy as np
import plotly.express as px
import polars as pl
import streamlit as st
from joblib import load
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

from util.altair_charts import (
    songs_by_bitrate,
    songs_per_year,
    top_5_genre_count_per_avg_bitrate,
    top_10_avg_durations_by_genre,
    top_10_genres_chart,
    top_10_most_played_songs,
)
from util.data_imputation import data_imputation
from util.genre_prediction import pick_random_song
from util.on_change import update_numin, update_slider
from util.scaler import scale

# "st.session_state object:", st.session_state

st.markdown(
    """
# Data Science for Music :guitar: :notes:
## Hypothetical case study

A music streaming company approached us to help them estimate analyze their
music library looking for patterns, also to figure out how much space they
would require to store their music files if they offered them to their users
in different bit rates, and finally they also wanted a to be able to classify
new songs in their library by to their genre automatically.

They provided us an export of their music files metadata, an export of how
many times their songs are played (also called scrobbles), and a few sample
songs of the genres they have in their library.
"""
)

# Import dataset

original_df = pl.read_csv('data/music_library_export.csv',
                          encoding='utf8-lossy')

# Data imputation

songs_df = data_imputation(original_df)

st.markdown(
    """
### Dataset 1 - Music Files Metadata

Here we can see the details of their songs' metadata included in their export.
""")

# Display original data's columns

st.dataframe(songs_df)

st.markdown(
    """
### Visualization
Now let's see a few charts we created for them to better understand their data.

Here's a chart of the top 10 genres with most songs in their dataset, we can
see the genre with most songs is 'Indie'.
"""
)

st.altair_chart(top_10_genres_chart(songs_df), theme=None)

st.markdown(
    """
Now here's a chart of the top 10 genres with the longest average song duration,
we can see the genre with most duration is 'Hardstyle'.
"""
)

st.altair_chart(top_10_avg_durations_by_genre(songs_df), theme=None)

st.markdown(
    """
Now here's a chart of the top 10 years with most songs, we can see the year
with most songs was 2008.
"""
)

st.altair_chart(songs_per_year(songs_df), theme=None)

st.markdown(
    """
Next, here's a chart of the top 5 most used bit rates, we can see most popular
one is 320 kb/s.
"""
)
st.altair_chart(songs_by_bitrate(songs_df), theme=None)

st.markdown(
    """
Now a chart of the top 5 genre count per average bit rate, we can see most
popular one is again 320 kb/s.
"""
)
base = top_5_genre_count_per_avg_bitrate(songs_df)
c1 = base.mark_arc(innerRadius=30, strokeWidth=0)
c2 = base.mark_text(radiusOffset=10).encode(
        text='Genre'
)
st.altair_chart(c1 + c2, theme=None)

# Import scrobbles dataset

scrobbles_df = pl.read_csv("data/lastfm-scrobbles-edchapa.csv")

# Data imputation

scrobbles_df.drop_in_place("uts")
scrobbles_df.drop_in_place("artist_mbid")
scrobbles_df.drop_in_place("album_mbid")
scrobbles_df.drop_in_place("track_mbid")

st.markdown(
    """
### Dataset 2 - Scrobbles

With the export of how many times their songs were played, we obtained the
following char of their top 10 most played songs.
"""
)

st.altair_chart(top_10_most_played_songs(scrobbles_df), theme=None)

st.markdown(
    """
### Storage Size Prediction
As we mentioned before, the music streaming company needed to figure out how
much space they would require to store their music files if they offered them
to their users in different bit rates.

For that, we will use their first dataset, but let's strip it down to their
songs' duration, bit rate, and size. Let's also see how their current total
storage size looks like."""
)

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
st.write('Current total storage size (GBs):',
         round(total_original_size.to_numpy()[0][0]/(1024*1024*1024), 2))

st.markdown(
    """
Now let's see an interactive 3D chart of the 'Size', 'Total Time' and
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
By manipulating the previous 3D chart, we can see that the size of a song
increases proportionally to its duration and its bit rate. The song with the
biggest size (represented by the yellow dot) also has the longest duration of
all songs, and the blue dots have the smallest size because they also have
either short durations and/or low bit rates.

This means we can predict the size of a song by fitting a linear regression
model using its duration and bit rate.
""")

st.sidebar.title("Music library storage size prediction :notes:")

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
### Modeling Part 1
After creating a linear regression model, now we can play with it and see how
the predicted total storage size will update automatically depending on the
selected bit rate by using the "Bit Rate (kb/s)" slider or the input field on
the left sidebar, so the company can make the best decision according to their
budget.
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

st.write("Predicted total storage size (GBs): ",
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

st.markdown(
    """
###
And now, here's how our model predicts all possible values within the ranges of
our dataset, from the minimum to the maximum values of the durations and bit
rates.
"""
)

# Generate 100 values between minimum and maximum durations and bitrates

min_bit_rate = songs_df.select(pl.col('Bit Rate').min()).to_numpy()[0][0]
max_bit_rate = songs_df.select(pl.col('Bit Rate').max()).to_numpy()[0][0]
min_duration = int(
    songs_df.select(pl.col('Total Time').min()).to_numpy()[0][0]/1000)
max_duration = int(
    songs_df.select(pl.col('Total Time').max()).to_numpy()[0][0]/1000)
bitrate_vals = np.linspace(min_bit_rate, max_bit_rate, 100)
duration_vals = np.linspace(min_duration, max_duration, 100)

# Create a grid combining all values

xy = np.meshgrid(duration_vals, bitrate_vals)
zz = np.array(list(zip(*(x.flat for x in xy))))

# Create dataframe with new values

plot_values = pl.DataFrame(zz).rename({"column_0": "Total Time", "column_1": "Bit Rate"})

# Scale Total Time

scaled_plot_values = scale(plot_values)

# predict new Size values and store them in a new DataFrame

predicted_y = model.predict(scaled_plot_values)

prediction_y_df = pl.DataFrame({"Size": np.concatenate(predicted_y)}).select(
    pl.col("Size").cast(pl.Int64)
)

# Remove negative Sizes

data = plot_values.with_columns(
    prediction_y_df).rename({'Total Time': 'Duration'}).select(
    pl.col('Duration').filter(pl.col('Size') > 0),
    pl.col('Bit Rate').filter(pl.col('Size') > 0),
    pl.col('Size').filter(pl.col('Size') > 0)
)

# Plot result

fig_3 = px.scatter_3d(
    data,
    x='Duration',
    y='Bit Rate',
    z='Size',
    color='Size'
)
fig_3.update_layout(scene=dict(camera=dict(eye=dict(x=1.5, y=1, z=1.5))))

st.plotly_chart(fig_3, theme=None)

# Music Genre Classification

st.markdown(
    """
### Dataset 3 - Music Genre Classification Model
Finally, the music streaming company wanted to automatically classify new
songs in their library according to their genre.

Our first step was extracting the audio features of the sample songs, here's
how they look like after extraction.
"""
)

# Import dataset

genres_df = pl.read_csv('data/features_3_sec.csv')

st.dataframe(genres_df)

# Export labels into a list

labels = genres_df[['label']].unique(maintain_order=True).to_series().to_list()

# Data imputation

genres_df.drop_in_place('length')
genres_df.drop_in_place('filename')

# Separate variables
labels_ds = genres_df.select(
    pl.col('label').cast(pl.Categorical).to_physical()).to_numpy().squeeze()
_ = genres_df.drop_in_place('label')

# Create StandardScaler

standard_scaler = StandardScaler()
scaled_features_ds = standard_scaler.fit_transform(genres_df.to_numpy())

# Split our dataset

total_song_count = len(scaled_features_ds)
training_count = int(total_song_count * .60)
test_count = int(total_song_count * .20)
validation_count = int(total_song_count * .20)
training_x, rest_x, training_y, rest_y = train_test_split(
    scaled_features_ds,
    labels_ds,
    train_size=training_count)
testing_x, validation_x, testing_y, validation_y = train_test_split(
    rest_x, rest_y, train_size=test_count)

# Load our model and its training history

history = np.load('models/training_history.npy', allow_pickle='TRUE').item()
model = load_model('models/genre_prediction.h5')

st.markdown(
    """
### Modeling Part 2
We built a genre classification neural network model and trained it using the
sample songs of each genre.

Our model has 1 Dense input layer, 1 hidden Dense layer, and 1 Dense output
layer, with 2 hidden Dropout layers in-between to prevent overfitting.
"""
)

model.summary(print_fn=lambda x: st.text(x))

st.markdown(
    """
We trained our model for 20 epochs, here's the model's Training History chart.
"""
)

# Plot our model's training history

accuracy = history.history['accuracy']
epochs = np.arange(len(accuracy))

fig = px.line(
    x=epochs,
    y=accuracy,
    title='Training History',
    labels={'x': 'Epochs', 'y': 'Accuracy'}
)

st.plotly_chart(fig)

# Plot the Loss History

st.markdown(
    """
And here's the model's Loss History chart.
"""
)

loss = history.history['loss']
epochs2 = np.arange(len(loss))

fig2 = px.line(
    x=epochs2,
    y=loss,
    title='Loss',
    labels={'x': 'Epochs', 'y': 'Loss'}
)

st.plotly_chart(fig2)

st.markdown(
    """
### Testing our model
After building our model, we wanted to see how well it was predicting the
sample songs' genre.

Here we can see a confusion matrix of the predicted genres vs the true genres
of the validation sample songs.
"""
)

# Evaluate vs validation dataset

model.evaluate(validation_x, validation_y, batch_size=128)
predictions_y = np.argmax(model.predict(validation_x, batch_size=128), axis=1)
validation_y_labels = []
predictions_y_labels = []
for val in validation_y:
    validation_y_labels.append(labels[val])
for val in predictions_y:
    predictions_y_labels.append(labels[val])
disp = ConfusionMatrixDisplay.from_predictions(validation_y_labels,
                                               predictions_y_labels)
disp.ax_.set_xticklabels(disp.ax_.get_xticklabels(), rotation=45)
st.pyplot(disp.figure_)

st.markdown(
    """
And here's the one for the testing sample songs.
"""
)

# Evaluate vs testing dataset

model.evaluate(testing_x, testing_y, batch_size=128)
predictions_y = np.argmax(model.predict(testing_x, batch_size=128), axis=1)
testing_y_labels = []
predictions_y_labels = []
for val in testing_y:
    testing_y_labels.append(labels[val])
for val in predictions_y:
    predictions_y_labels.append(labels[val])
disp = ConfusionMatrixDisplay.from_predictions(testing_y_labels,
                                               predictions_y_labels)
disp.ax_.set_xticklabels(disp.ax_.get_xticklabels(), rotation=45)
st.pyplot(disp.figure_)

st.markdown(
    """
Finally, by pressing the Predict button on the left sidebar we can test our
model and predict the genre of a randomly selected song.
""")

st.sidebar.title("Music genre prediction :guitar:")

st.sidebar.button("Predict", on_click=pick_random_song(model, standard_scaler,
                                                       labels))
