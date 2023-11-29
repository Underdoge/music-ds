""" Module with update functions for streamlit widgets."""
import streamlit as st


def update_slider() -> None:
    """ Update the value of the sidebar slider when the input changes."""
    st.session_state.slider = st.session_state.numeric


def update_numin() -> None:
    """ Update the value of the input when the slider changes."""
    st.session_state.numeric = st.session_state.slider
