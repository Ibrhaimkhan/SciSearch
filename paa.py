import streamlit as st
import pandas as pd

st.write("# SciSearch 🔬")
st.markdown(":red[This text is red.]")
st.markdown(":blue-background[This text has a blue background.]")

st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4CAF50; /* Green background */
        color: white; /* White text */
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.button("Click me")