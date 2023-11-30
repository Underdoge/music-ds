# music-ds
## Data Science for Music 🎸🎶
## Final Project for the Código Facilito's 2023 Data Science Bootcamp
<img width="800" alt="image" src="https://github.com/Underdoge/music-ds/assets/12192446/6009b949-9d78-4ab9-b64f-0665644e862b">

# Installing the Streamlit App
Open up a Terminal (macOS/Linux) or PowerShell (Windows) and enter the following commands:
### Cloning the repository
```sh
git clone https://github.com/underdoge/music-ds

cd music-ds
```
### Creating the virtual environment
```sh
python -m venv venv
```
### Activating the virtual environment on macOS / Linux
```sh
source venv/bin/activate
```
### Activating the virtual environment on Windows (PowerShell)
```powershell
.\venv\Scripts\Activate.ps1
```
### Installing requirements
```sh
pip install -r requirements.txt
```
#
# Running the Streamlit App
### Running the program on macOS / Linux
```sh
streamlit run app.py
```
#
# Hosted Streamlit App
A hosted version of the app can be found on Streamlit [here](https://music-ds.streamlit.app).
#
# Requirements
- Python 3.11 or greater
- Git (to clone the repo)
#
# Dataset Sources
- My Winamp music library's Media Library Export - can be found under data/music_library_export.xml
- My [Last.fm account](https://www.last.fm/user/edchapa)'s scrobbles (time series of when each song was played) extracted from [here](https://lastfm.ghan.nl/export/) - can be found under data/lastfm-scrobbles-edchapa.csv
- [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) from Kaggle.
#
# Not Supported
At the moment, there's a [bug while loading h5 keras models on Windows created on macOS/Linux](https://github.com/keras-team/keras/issues/18528). You may encounter this bug when running the Streamlit app directly on Windows right after cloning the repo if you don't run the Jupyter Notekbook first. The workaround is to run the Streamlit app from [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install), or running the Jupyter Notebook on Windows first (which will create new model files that don't have the issue on Windows) before running the Streamlit app.
