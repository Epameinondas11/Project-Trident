import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import cosine_similarity, euclidean_distances

# ============================================
# ⚙️ PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title = "🔱 Project Trident",
    page_icon = "🔱",
    layout = "wide",
    initial_sidebar_state = "collapsed"
)

# ============================================
# 🎨 CUSTOM CSS STYLING
# ============================================

    