import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from sonar import (
    load_and_prep_data,
    find_similar_players_gui,
    classify_player_role,
    get_weights_by_role,
    league_weights
)


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


def local_css():
    st.markdown(
        """
        <style>
        /* --- Global Styles --- */
        .main {
            background-color: #0e1117;
        }
        
        /* ===== TYPOGRAPHY ===== */
    h1, h2, h3 {
        color: #FAFAFA !important;
        font-family: 'Inter', sans-serif;
    }
    
    p, label {
        color: #A0A0A0;
    }
    
    /* ===== PLAYER CARD ===== */
    .player-card {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 24px;
        border-radius: 12px;
        border-left: 4px solid #00D9FF;
        margin: 16px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
     .player-card h2 {
        color: #FAFAFA;
        margin: 0;
        font-size: 28px;
        font-weight: bold;
    }
    
    .player-card p {
        color: #A0A0A0;
        margin: 8px 0;
        font-size: 16px;
    }
    
    .role-badge {
        display: inline-block;
        background: #00D9FF;
        color: #0E1117;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
        margin-top: 8px;
    }
    
    /* ===== PLAYER GRID CARD (Browse) ===== */
    .player-grid-card {
        background: #1E1E1E;
        padding: 16px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #2D2D2D;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .player-grid-card:hover {
        border-color: #00D9FF;
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0, 217, 255, 0.2);
    }
    
    .player-grid-card h4 {
        color: #FAFAFA;
        margin: 0 0 8px 0;
        font-size: 18px;
    }
    
    .player-grid-card p {
        color: #A0A0A0;
        margin: 4px 0;
        font-size: 14px;
    }
    
    /* ===== STATS BOX ===== */
    .stat-box {
        background: #1E1E1E;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #2D2D2D;
    }
    
    .stat-box h3 {
        color: #00D9FF;
        margin: 0;
        font-size: 32px;
        font-weight: bold;
    }
    
    .stat-box p {
        color: #A0A0A0;
        margin: 8px 0 0 0;
        font-size: 14px;
    }
    
    /* ===== SIMILARITY BADGES ===== */
    .sim-high {
        color: #51CF66;
        font-weight: bold;
        font-size: 16px;
    }
    
    .sim-med {
        color: #FFD43B;
        font-weight: bold;
        font-size: 16px;
    }
    
    .sim-low {
        color: #FF6B6B;
        font-weight: bold;
        font-size: 16px;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #00D9FF 0%, #0099CC 100%);
        color: #0E1117;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00F0FF 0%, #00B8E6 100%);
        box-shadow: 0 4px 12px rgba(0, 217, 255, 0.4);
        transform: translateY(-2px);
    }
    
    /* ===== DIVIDER ===== */
    hr {
        border: none;
        border-top: 1px solid #2D2D2D;
        margin: 32px 0;
    }
    
    /* ===== DATAFRAME STYLING ===== */
    .dataframe {
        background-color: #1E1E1E !important;
        color: #FAFAFA !important;
    }
    
    .dataframe th {
        background-color: #2D2D2D !important;
        color: #00D9FF !important;
        font-weight: bold;
    }
    
    .dataframe td {
        background-color: #1E1E1E !important;
        color: #FAFAFA !important;
    }
    
    /* ===== METRICS ===== */
    [data-testid="stMetricValue"] {
        color: #00D9FF !important;
        font-size: 32px !important;
        font-weight: bold !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #A0A0A0 !important;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1E1E1E;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        color: #A0A0A0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2D2D2D;
        color: #00D9FF !important;
    }
    
    </style>
    """, unsafe_allow_html=True)
    

# ============================================
# 💾 DATA LOADING (με Streamlit caching)
# ============================================


@st.cache_data
def load_cached_data():
    
    """
    Φορτώνει τα δεδομένα με caching για performance.
    """
    
    return load_and_prep_data('perfect_merge.csv')


# ============================================
# 🎯 UI COMPONENTS
# ============================================


def render_player_card(player):
    """
    Εμφανίζει player profile card με stats.
    """
    league = player.get('League_Clean', player.get('Comp', 'Unknown'))
    
    st.markdown(f"""
    <div class="player-card">
        <h2>{player['Player']}</h2>
        <p><strong>🏟️ {player['Squad']}</strong> | 👤 Age: {int(player['Age'])} | 📍 {player['Pos']}</p>
        <p>🌍 {league} | ⏱️ {int(player['Min'])} minutes played</p>
        <span class="role-badge">{player['Role']}</span>
    </div>
    """, unsafe_allow_html=True)
    
    
def render_stats_metrics(player):
    
    """
    Εμφανίζει key stats σε metric format.
    """
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("⚽ Goals", int(player['Gls']))
    
    with col2:
        st.metric("🎯 Assists", int(player['Ast']))
    
    with col3:
        st.metric("📊 G/Sh", f"{player['G/Sh']:.2f}")
    
    with col4:
        st.metric("🎪 SoT%", f"{player['SoT%']:.1f}%")
    
    with col5:
        st.metric("🔫 Sh/90", f"{player['Sh/90']:.1f}")
        
        
def get_similarity_badge(score):
    
    """
    Επιστρέφει HTML badge ανάλογα με το similarity score.
    """
    
    if score >= 85:
        return f'<span class="sim-high">🟢 {score:.1f}%</span>'
    elif score >= 70:
        return f'<span class="sim-med">🟡 {score:.1f}%</span>'
    else:
        return f'<span class="sim-low">🔴 {score:.1f}%</span>'
    

def render_results_table(results):
    
    """
    Εμφανίζει τα αποτελέσματα σε styled table.
    """
    
    if results is None or len(results) == 0:
        st.warning("No similar players found.")
        return
    
    # Προετοιμασία δεδομένων
    display_df = results[['Player', 'Squad', 'Role', 'Gls', 'Ast', 'G/Sh', 'SoT%', 'Similarity_Score']].copy()
    
    # Μορφοποίηση
    display_df['Gls'] = display_df['Gls'].astype(int)
    display_df['Ast'] = display_df['Ast'].astype(int)
    display_df['G/Sh'] = display_df['G/Sh'].apply(lambda x: f"{x:.2f}")
    display_df['SoT%'] = display_df['SoT%'].apply(lambda x: f"{x:.1f}%")
    display_df['Similarity'] = display_df['Similarity_Score'].apply(lambda x: f"{x:.1f}%")
    
    display_df = display_df.drop('Similarity_Score', axis=1) # Δεν θέλουμε να εμφανίσουμε την raw score στήλη
    
    # Εμφάνιση
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        hide_index=True        
    )
    
def render_radar_chart(target, similar_players):
    
    """
    Δημιουργεί radar chart για σύγκριση παικτών.
    """
    
    if similar_players is None or len(similar_players) == 0:
        st.warning("Not enough data for radar chart")
        return
    
    # Metrics για radar
    metrics = ['Gls_Adj', 'Ast_Adj', 'Sh/90', 'SoT%', 'G/Sh']
    
    # Ελέγχουμε αν υπάρχουν οι στήλες
    available_metrics = [m for m in metrics if m in target.index]
    
    if len(available_metrics) < 3:
        st.warning("Not enough metrics available for radar chart")
        return
    
    # Plotly radar chart
    fig = go.Figure()
    
    colors = ['#00D9FF', '#51CF66', '#FFD43B', '#FF6B6B']
    
    # Προσθήκη target παίκτη
    fig.add_trace(go.Scatterpolar(
        r=[target[m] for m in available_metrics],
        theta=available_metrics,
        fill='toself',
        name=target['Player'],
        line_color=colors[0],
        opacity=0.7
    ))
    
    # Top 3 παρόμοιοι παίκτες
    for idx, (_, row) in enumerate(similar_players.head(3).iterrows()):
        if idx >= 3:
            break
        fig.add_trace(go.Scatterpolar(
            r=[row[m] for m in available_metrics],
            theta=available_metrics,
            fill='toself',
            name=row['Player'],
            line_color=colors[idx + 1],
            opacity=0.6
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                showgrid=True,
                gridcolor='#2D2D2D',
            ),
            bgcolor='#1E1E1E'
        ),
        showlegend=True,
        template='plotly_dark',
        height=500,
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font=dict(color='#FAFAFA')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    

# ============================================
# 📄 PAGE: HOME
# ============================================


def page_home():
    
    """
    Home page με search bar και role browsing.
    """
    
    # Header (Compact)
    st.markdown("<h1 style='margin-bottom: 0px;'>🔱 PROJECT TRIDENT</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0px; color: #A0A0A0;'>Player Similarity Search Engine</h3>", unsafe_allow_html=True)
    st.caption("⚽ Top 5 Leagues 2024/25")
    
    # Compact Description (2 columns)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='font-size: 14px;'>
        <p style='margin-bottom: 8px;'>Find football players with <strong>similar playing styles</strong> across Europe's elite leagues.</p>
        <p style='margin-bottom: 4px;'><strong>Perfect for:</strong></p>
        <ul style='margin-top: 0px; font-size: 13px;'>
            <li>🔍 Scouts finding transfer targets</li>
            <li>📊 Analysts comparing player profiles</li>
            <li>⚽ Fans exploring tactical alternatives</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='font-size: 14px;'>
        <p style='margin-bottom: 4px;'><strong>How it works:</strong></p>
        <ul style='margin-top: 0px; font-size: 13px;'>
            <li>Advanced similarity algorithms (Cosine/Euclidean)</li>
            <li>Adjusted for league difficulty</li>
            <li>8 distinct player roles</li>
            <li>5 key performance metrics</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Search Section (More Prominent)
    st.markdown("### 🔎 Search Player by Name")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        search_query = st.text_input(
            "Player Name",
            placeholder="e.g., Haaland, Mbappe, Kane...",
            key="search_home",
            label_visibility="collapsed"
        )
        
    with col2:
        search_btn = st.button("🚀 Search", type="primary", use_container_width=True)
        
    if search_btn and search_query:
        st.session_state.page = "search"
        st.session_state.query = search_query
        st.rerun()
        
    st.markdown("---")
    
    # Role Browsing Section
    st.markdown("### 🎯 Or Browse by Role")
    
    roles = [
        "💀 Killer Striker",
        "🎯 Elite Striker", 
        "⚽ Striker",
        "🔗 Support Striker",
        "🚀 Winger / Inside Forward",
        "⚡ Winger (Attacking)",
        "👻 Shadow Striker / Creator",
        "🏹 Supporting Winger"
    ]
    
    # Grid layout (4 columns)
    cols = st.columns(4)
    
    for idx, role in enumerate(roles):
        with cols[idx % 4]:
            if st.button(role, key=f"role_{idx}", use_container_width=True):
                st.session_state.page = "browse"
                st.session_state.selected_role = role
                st.rerun()
                
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📊 Database Overview")
    
    df = load_cached_data()
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📋 Players Analyzed", len(df))
        
        with col2:
            st.metric("🌍 Leagues Covered", 5)
        
        with col3:
            st.metric("🎭 Role Categories", 8)
        
        with col4:
            st.metric("⚽ Total Goals", int(df['Gls'].sum()))


# ============================================
# 📄 PAGE: SEARCH RESULTS
# ============================================


def page_search_results():

    """
    Εμφανίζει τα αποτελέσματα αναζήτησης με similarity scores.
    """

    # Back button
    if st.button("🔙 Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.title("🔎 Search Results")

    # Load data
    df = load_cached_data()

    if df is None:
        st.error("❌ Failed to load data.")
        return
    
    query = st.session_state.get("query", "")

    # Εύρεση παίκτη(ες)
    matches = df[df['Player'].str.contains(query, case=False, na=False)]

    if len(matches) == 0:
        st.error(f"❌ No players found matching '{query}'.")
        st.info("💡 Try searching by last name (e.g., 'Haaland' instead of  'Erling Haaland')")
        return
    
    elif len(matches) > 1:
        st.warning(f"👥 Found {len(matches)} players matching '{query}'. Please select:")

        players_options = [
            f"{row['Player']} ({row['Squad']}, {row['Age']})" for _, row in matches.iterrows()
        ]
        
        selected = st.selectbox("Choose Player", players_options)
        selected_idx = players_options.index(selected)
        target = matches.iloc[selected_idx]
    else:
        target = matches.iloc[0]

    st.markdown("---")

    # Εμφάνιση player card
    st.subheader("🎭 Target Player Profile")

    col1, col2 = st.columns([1, 2])

    with col1:
        render_player_card(target)
    
    with col2:
        st.markdown(f"### 📊 Key Stats for {target['Player']}")
        render_stats_metrics(target)

    st.markdown("---")

    # Ρυθμίσεις αλγορίθμου
    st.subheader("⚙️ Search Configuration")

    col1, col2 = st.columns([2, 1])

    with col1:
        algorithm = st.selectbox(
            "Similarity Algorithm",
            ["cosine", "euclidean"],
            format_func=lambda x: "🎯 Cosine Similarity" if x == "cosine" else "📏 Euclidean Distance"
        )

    with col2:
        top_n = st.slider("Number of Similar Players", 5, 20, 10)

    # Εύρεση παρόμοιων παικτών
    with st.spinner("🔍 Searching for similar players..."):
        results = find_similar_players_gui(
            df=df,
            target_player=target,
            algorithm=algorithm,
            n_neighbors=top_n
        )

    if results is None or len(results) == 0:
        st.warning("❌ No similar players found.")
        return
    
    st.markdown("---")

    # Εμφάνιση αποτελεσμάτων
    st.subheader(f"📋 Similar Players ({len(results)}) Results")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Table View", "📈 Visual Comparison", "🔍 Detailed Stats"])

    with tab1:
        render_results_table(results)

        # Βutton για εξαγωγή αποτελεσμάτων σε CSV
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Results as CSV",
            data=csv,
            file_name=f"similar_players_{target['Player'].replace(' ', '_')}.csv",
            mime='text/csv',
            use_container_width=True
        )

    with tab2:
        st.markdown("#### 🕸️ Player Comparison Radar")
        render_radar_chart(target, results)

        # Bar chart για similarity scores
        st.markdown("#### 📊 Similarity Scores")

        fig_bar = px.bar(
            results.head(10),
            x='Player',
            y='Similarity_Score',
            color='Similarity_Score',
            color_continuous_scale=['#FF6B6B', '#FFD43B', '#51CF66'],
            labels={'Similarity_Score': 'Similarity (%)'},
            title="Top 10 Similar Players"
        )

        fig_bar.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117',
            font=dict(color='#FAFAFA'),
            xaxis_tickangle=-45
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.markdown("#### 📋 Detailed Players Profiles")

        for idx, (_, row) in enumerate(results.head(10).iterrows()):
            with st.expander(f"#{idx + 1} {row['Player']} - {row['Squad']} ({row['Similarity_Score']:.1f}% similar)"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("**Basic Info:**")
                    st.write(f"- Role: {row['Role']}")
                    st.write(f"- Age: {int(row['Age'])}")
                    st.write(f"- Position: {row['Pos']}")
                    st.write(f"- Minutes: {int(row['Min'])}")

                with col2:
                    st .write("**Goals & Assists:**")
                    st.write(f"- Goals: {int(row['Gls'])}")
                    st.write(f"- Assists: {int(row['Ast'])}")
                    st.write(f"- G+A: {int(row['Gls'] + row['Ast'])}")
                    st.write(f"- Non-Penalty : {int(row['G-PK'])}")

                with col3:
                    st.write("**Shooting Stats:**")
                    st.write(f"- Shots/90: {row['Sh/90']:.1f}")
                    st.write(f"- SoT%: {row['SoT%']:.1f}%")
                    st.write(f"- G/Sh: {row['G/Sh']:.2f}")
                    if 'G/SoT' in row.index:
                        st.write(f"- G/SoT: {row['G/SoT']:.2f}")


# ============================================
# 📄 PAGE: BROWSE BY ROLE
# ============================================


def page_browse_role():

    """
    Εμφανίζει παίκτες ανά ρόλο με δυνατότητα φιλτραρίσματος.
    """

    # Βack button
    if st.button("🔙 Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    role = st.session_state.get("selected_role", '')
    st.title(f"{role}")

    # Φορτώνουμε τα δεδομένα
    df = load_cached_data()

    if df is None:
        st.error("❌ Failed to load data.")
        return
    
    # Φιλτράρουμε για τον επιλεγμένο ρόλο
    filtered = df[df['Role'] == role]

    st.caption(f"📊 {len(filtered)} players found")

    st.markdown("---")

    # Φίλτρα
    st.subheader("🔧 Filters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        leagues = ['Top 5 Leagues']
        if 'League_Clean' in df.columns:
            leagues += sorted(df['League_Clean'].unique().tolist())
        elif 'Comp' in df.columns:
            leagues += sorted(df['Comp'].unique().tolist())

        league_filter = st.selectbox("🌍 League", leagues)

    with col2:
        age_range = st.slider("👤 Age Range", 16, 40, (18, 35))

    with col3:
        sort_options = {
            "Goals (Gls)": "Gls",
            "Assists (Ast)": "Ast",
            "Goals per Shot (G/Sh)": "G/Sh",
            "Shot Accuracy (SoT%)": "SoT%",
            "Shots per 90 (Sh/90)": "Sh/90"
        }
        sort_by = st.selectbox("📊 Sort By", list(sort_options.keys()))

    with col4:
        min_minutes = st.number_input("⏱️ Min Minutes Played", 0, 3000, 450, step=50)
        
    # Εφαρμογή φίλτρων
    if league_filter != 'Top 5 Leagues':
        if 'League_Clean' in filtered.columns:
            filtered = filtered[filtered['League_Clean'] == league_filter]
        elif 'Comp' in filtered.columns:
            filtered = filtered[filtered['Comp'] == league_filter]
            
    filtered = filtered[
        (filtered['Age'] >= age_range[0]) &
        (filtered['Age'] <= age_range[1]) &
        (filtered['Min'] >= min_minutes)
    ]
    
    filtered = filtered.sort_values(by=sort_options[sort_by], ascending=False)
    
    st.markdown("---")
    
    # Εμφάνιση αποτελεσμάτων σε grid
    st.subheader(f"📋 {len(filtered)} Players")
    
    if len(filtered) == 0:
        st.warning("❌ No players found with the selected filters.")
        return
    
    # Grid layout (4 columns per row)
    cols_per_row = 4
    
    for i in range(0, len(filtered), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(filtered):
                player = filtered.iloc[i + j]
                
                with col:
                    st.markdown(f"""
                    <div class="player-grid-card">
                        <h4>{player['Player']}</h4>
                        <p>{player['Squad']}</p>
                        <p><strong>{int(player['Gls'])}G</strong> | {int(player['Ast'])}A</p>
                        <p>{player['SoT%']:.1f}% SoT</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("🔎 View", key=f"view_{i+j}", use_container_width=True):
                        st.session_state.page = "search"
                        st.session_state.query = player['Player']
                        st.rerun()
                        

# ============================================
# 🚀 MAIN APP ROUTER
# ============================================



def main():
    """
    Main app controller με page routing.
    """
    # Load CSS
    local_css()
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    # Route to appropriate page
    if st.session_state.page == "home":
        page_home()
    elif st.session_state.page == "search":
        page_search_results()
    elif st.session_state.page == "browse":
        page_browse_role()
        
        

# ============================================
# 🎬 RUN APP
# ============================================


if __name__ == "__main__":
    main()