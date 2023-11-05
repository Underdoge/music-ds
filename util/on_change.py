import streamlit as st


def update_slider():
    st.session_state.slider = st.session_state.numeric


def update_numin():

    st.session_state.numeric = st.session_state.slider
