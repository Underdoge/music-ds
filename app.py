import streamlit as st
import polars as pl
from joblib import load
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import RobustScaler



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
        lambda x: "Electronic Pop" if x in ["Pop Electronica", "Electronica / Pop"] else x)
    .map_elements(
        lambda x: "Indie" if x == "indie" else x)
    .map_elements(
        lambda x: "Indie Rock" if x in ["Rock/Indie", "Indie/Rock", "General Indie Rock"] else x)
    .map_elements(
        lambda x: "Miscellaneous" if x == "misc" else x)
    .map_elements(
        lambda x: "Soundtrack" if x in ["soundtrack", "Banda sonora"] else x)
    .map_elements(
        lambda x: "Thrash Metal" if x == "Thrash Metal" else x)
    .map_elements(
        lambda x: "Alt Rock" if x in ["Alt. Rock", "Alternative Rock", "Rock alternativo",
                                      "Alternative, Rock", "General Alternative Rock"] else x)
    .map_elements(
        lambda x: "Brit Pop" if x == "Brit-pop" else x)
    .map_elements(
        lambda x: "Pop Rock" if x in ["Pop/Rock", "Pop/Rock 2000's"] else x)
    .map_elements(
        lambda x: "Pop" if x == "General Pop" else x)
    .map_elements(
        lambda x: "Folk" if x == "General Folk" else x)
    .map_elements(
        lambda x: "Rock" if x in ["General Rock", "Rock En General", "Rock en general", "Rock @",
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
        lambda x: None if x in ["genre", "default", ".", "(255)", "Other"] else x)
    .map_elements(
        lambda x: "Unclassifiable" if x == "General Unclassifiable" else x)
    .map_elements(
        lambda x: "Soft Rock / Alternative Folk / Folk / Rock" if x == "soft rock/alternative folk/folk/rock" else x)
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

st.markdown(
    """
### Original Data

Here we can see their original songs duration, size and bit \
rate, and their total storage size requirement.
""")

original_data = songs_df.select(
    (pl.col('Total Time')/1000).alias('Duration (seconds)'),
    pl.col('Bit Rate').alias('Bit Rate (kb/s)')).with_columns(
                           songs_df.select(pl.col('Size').alias(
                               'Size (bytes)'))
)
st.dataframe(original_data, hide_index=True)

total_original_size = songs_df.select(
    pl.col('Size').sum())
st.write('Total storage size (GBs):',
         round(total_original_size.to_numpy()[0][0]/(1024*1024*1024), 2))

st.sidebar.title("Music storage size prediction :notes:")

bit_rate = st.sidebar.slider("Bit rate (kb/s)", 24, 2822, 320)

st.markdown(
    """
### Storage Size Prediction
Here we can play with the model and see how the required storage will change \
depending on the selected bit rate.

Using the slider on the left, we can choose different bit rates and see their \
total predicted storage size, so the company can make the best decision \
according to their budget.
"""
)

# Loading the model we created before

model = load('models/music_size.joblib')

songs_independent_vars = songs_df.select((pl.col('Total Time')/1000),
                                         pl.col('Bit Rate'))
st.write("New bit rate (kb/s):", bit_rate)
new_df = songs_independent_vars.with_columns(
    pl.col('Bit Rate').map_elements(lambda x: bit_rate)
)

predicted_size = model.predict(new_df)

st.write("Total predicted storage size (GBs): ",
         round(predicted_size.sum()/(1024*1024*1024), 2))
