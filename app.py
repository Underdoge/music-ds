"""
This is a streamlit app that will showcase an hypothetical use case for the
'music_size.joblib' and 'genre_prediction.joblib' models under the 'models'
folder.
"""
import streamlit as st
import polars as pl
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from joblib import load
from util.on_change import update_slider, update_numin
from util.scaler import scale
from util.altair_charts import top_10_genres_chart
from util.altair_charts import top_10_avg_durations_by_genre
from util.altair_charts import songs_per_year, songs_by_bitrate
from util.altair_charts import top_5_genre_count_per_avg_bitrate
from util.altair_charts import top_10_most_played_songs
from util.data_imputation import data_imputation
from util.genre_prediction import pick_random_song
import plotly.express as px
import numpy as np

# "st.session_state object:", st.session_state

st.markdown(
    """
# Data Science for Music :notes:
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

original_df = pl.read_csv('data/music_library_export.csv')

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
c1 = base.mark_text(radiusOffset=10).encode(
        text='Genre'
)
st.altair_chart(base + c1, theme=None)

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

# Music Genre Classification

st.markdown(
    """
### Modeling Part 2
Finally, the music streaming company wanted to automatically classify new
songs in their library according to their genre.

For that, we built a genre classification neural network model using their
sample songs of each music genre they have in their library.

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

# Load our model and its training history using np.load

history = np.load('models/training_history.npy', allow_pickle='TRUE').item()
model = load_model('models/genre_prediction.keras')

st.markdown(
    """
### Music Genre Prediction Model
Here's the model's Training History chart, we trained it for 6 epochs.
"""
)

# Plot our model's training history

accuracy = history['accuracy']
epochs = np.arange(len(accuracy))

fig = px.line(
    x=epochs,
    y=accuracy,
    title='Training History',
    labels=dict(x='Epochs', y='Accuracy')
)

st.plotly_chart(fig)

st.markdown(
    """
After building our model, we wanted to see how well it was predicting the
sample songs' genre.

Here we can see a confusion matrix of the validation sample songs' predicted
genre vs their true genre.
"""
)

# Evaluate vs validation dataset

model.evaluate(validation_x, validation_y, batch_size=128)
predictions_y = np.argmax(model.predict(validation_x, batch_size=128), axis=1)
validation_confusion_matrix = confusion_matrix(validation_y, predictions_y)
validation_df = pl.DataFrame(validation_confusion_matrix,
                             schema=labels)
validation_df = pl.DataFrame(pl.Series("Genre", labels)).with_columns(
    validation_df
)
st.dataframe(validation_df, hide_index=True)
# st.altair_chart(confusion_matrix_display(validation_df))

st.markdown(
    """
And here's the one for the testing sample songs.
"""
)

# Evaluate vs testing dataset

model.evaluate(testing_x, testing_y, batch_size=128)
predictions_y = np.argmax(model.predict(testing_x, batch_size=128), axis=1)
testing_confusion_matrix = confusion_matrix(testing_y, predictions_y)
testing_df = pl.DataFrame(testing_confusion_matrix,
                          schema=labels)
testing_df = pl.DataFrame(pl.Series("Genre", labels)).with_columns(
    testing_df
)
st.dataframe(testing_df, hide_index=True)
# st.altair_chart(confusion_matrix_display(testing_df))


st.markdown(
    """
### Testing our model
Finally, here we can test our model by predicting the genre of a randomly
selected song by pressing the "Predict" button on the sidebar.
""")

st.sidebar.title("Music genre prediction :guitar:")

st.sidebar.button("Predict", on_click=pick_random_song(model, standard_scaler,
                                                       labels))
