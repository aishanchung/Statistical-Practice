import streamlit as st
from dashboard.data_loader import bundle
from dashboard.components.view import overview
overview(bundle(),st.session_state['choice'])
