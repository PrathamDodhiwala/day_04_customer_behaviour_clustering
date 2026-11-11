
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="BEHAVIOUR LAB | Customer Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# DESIGN SYSTEM
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #101310;
    --panel: #191e19;
    --panel2: #202720;
    --border: #303a30;
    --text: #edf1e9;
    --muted: #929b8e;
    --lime: #c5f36a;
    --lime-dark: #a6d34f;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Manrope', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4, p, label {
    font-family: 'Manrope', sans-serif;
}

h1 {
    font-weight: 800 !important;
    letter-spacing: -2px;
}

h2 {
    letter-spacing: -1px;
    font-weight: 750 !important;
}

h3 {
    font-weight: 700 !important;
}

[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 19px 20px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

[data-testid="stMetricValue"] {
    color: var(--lime) !important;
    font-weight: 800;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

.stButton > button,
.stDownloadButton > button {
    background: var(--lime);
    color: #101310;
    border: none;
    border-radius: 7px;
    font-weight: 800;
    min-height: 42px;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #d8ff8a;
    color: #101310;
    border: none;
}

.stTextInput input,
.stSelectbox div[data-baseweb="select"],
.stNumberInput input {
    background: var(--panel);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    color: var(--muted);
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    color: var(--lime) !important;
}

hr {
    border-color: var(--border);
}

.lab-topline {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 35px;
}

.lab-brand {
    font-family: 'DM Mono', monospace;
    color: var(--lime);
    font-size: 13px;
    letter-spacing: 2px;
}

.lab-status {
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 1px;
}

.hero {
    padding: 32px 0 38px 0;
}

.eyebrow {
    font-family: 'DM Mono', monospace;
    color: var(--lime);
    font-size: 11px;
    letter-spacing: 2px;
    margin-bottom: 16px;
}

.hero-title {
    font-size: clamp(38px, 6vw, 68px);
    font-weight: 800;
    letter-spacing: -4px;
    line-height: 1.03;
    color: var(--text);
}

.hero-title span {
    color: var(--lime);
}

.hero-copy {
    color: var(--muted);
    font-size: 15px;
    max-width: 650px;
    line-height: 1.8;
    margin-top: 20px;
}

.section-kicker {
    color: var(--lime);
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.section-heading {
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 5px;
}

.section-copy {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 20px;
}

.info-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    height: 100%;
}

.info-label {
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.info-value {
    font-size: 25px;
    font-weight: 800;
    color: var(--lime);
    margin-top: 10px;
}

.info-note {
    color: var(--muted);
    font-size: 12px;
    margin-top: 5px;
}

.cluster-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 10px;
}

.cluster-card strong {
    color: var(--lime);
    font-size: 20px;
}

.cluster-card small {
    color: var(--muted);
}

.footer {
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    padding-top: 25px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA GENERATION
# =========================================================

@st.cache_data
def generate_demo_data(n_per_group=120):
    rng = np.random.default_rng(42)

    personas = [
        [3, 4, 8, 1, 1, 0, 3, 1],
        [12, 12, 35, 8, 12, 4, 18, 1],
        [25, 22, 75, 18, 30, 12, 27, 2],
        [8, 7, 18, 3, 2, 1, 8, 7]
    ]

    columns = [
        "app_sessions",
        "avg_session_minutes",
        "content_views",
        "searches",
        "likes",
        "shares",
        "active_days",
        "support_requests"
    ]

    rows = []

    for persona in personas:
        for _ in range(n_per_group):
            values = [
                max(0, rng.normal(persona[0], 2.2)),
                max(0.5, rng.normal(persona[1], 2.5)),
                max(0, rng.normal(persona[2], 8)),
                max(0, rng.normal(persona[3], 2)),
                max(0, rng.normal(persona[4], 4)),
                max(0, rng.normal(persona[5], 2)),
                np.clip(rng.normal(persona[6], 3), 1, 30),
                max(0, rng.normal(persona[7], 1.5))
            ]

            rows.append(values)

    df = pd.DataFrame(rows, columns=columns).round(2)
    df.insert(
        0,
        "user_id",
        [f"USER-{i:04d}" for i in range(1, len(df) + 1)]
    )

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# =========================================================
# HELPERS
# =========================================================

def calculate_silhouette(data, labels):
    labels = np.asarray(labels)

    if len(np.unique(labels)) < 2:
        return None

    if len(np.unique(labels)) >= len(labels):
        return None

    try:
        return silhouette_score(data, labels)
    except Exception:
        return None


def cluster_count(labels):
    return len(set(labels) - {-1})


def create_pca(data, labels, ids):
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(data)

    visual = pd.DataFrame({
        "PC 1": coordinates[:, 0],
        "PC 2": coordinates[:, 1],
        "Cluster": [
            "Noise" if x == -1 else f"Group {x}"
            for x in labels
        ],
        "User ID": ids.astype(str)
    })

    variance = pca.explained_variance_ratio_.sum()

    return visual, variance


# =========================================================
# TOP BAR
# =========================================================

st.markdown("""
<div class="lab-topline">
    <div class="lab-brand">◈ BEHAVIOUR LAB</div>
    <div class="lab-status">DAY 04 / 60 &nbsp; • &nbsp; ML RESEARCH WORKSPACE</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">CUSTOMER INTELLIGENCE / UNSUPERVISED LEARNING</div>
    <div class="hero-title">
        Understand the<br>
        <span>patterns behind</span><br>
        behaviour.
    </div>
    <div class="hero-copy">
        Discover hidden behavioural segments using machine learning.
        Explore user engagement, compare clustering algorithms and
        investigate the structure of your customer data.
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# NAVIGATION
# =========================================================

page = st.radio(
    "Workspace navigation",
    [
        "01  Overview",
        "02  Cluster Studio",
        "03  Segment Explorer",
        "04  Data Export"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()


# =========================================================
# DATA SOURCE
# =========================================================

with st.sidebar:
    st.markdown("## ◈ Data controls")
    source = st.radio(
        "Dataset",
        ["Demo dataset", "Upload CSV"]
    )

    if source == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload behaviour data",
            type=["csv"]
        )
    else:
        uploaded = None

    st.divider()
    st.caption("BEHAVIOUR LAB / SETTINGS")
    st.caption("Your uploaded data is processed in this app session.")


if source == "Upload CSV":
    if uploaded is None:
        st.info("Open the sidebar and upload a CSV dataset to enter the workspace.")
        st.stop()

    try:
        df = pd.read_csv(uploaded)
    except Exception as error:
        st.error(f"Could not read the CSV: {error}")
        st.stop()
else:
    df = generate_demo_data()


if df.empty or len(df) < 3:
    st.error("The dataset must contain at least three rows.")
    st.stop()


# =========================================================
# DATA PREPARATION
# =========================================================

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_columns) < 2:
    st.error("Your dataset needs at least two numeric columns.")
    st.stop()

default_features = [
    c for c in numeric_columns
    if df[c].nunique(dropna=True) > 1
]

with st.sidebar:
    selected_features = st.multiselect(
        "Behaviour features",
        numeric_columns,
        default=default_features
    )

    algorithm = st.selectbox(
        "Clustering algorithm",
        ["K-Means", "DBSCAN", "Compare Both"]
    )

    k = st.slider("Number of K-Means groups", 2, 10, 4)
    eps = st.slider("DBSCAN epsilon", 0.1, 3.0, 0.8, 0.1)
    min_samples = st.slider("DBSCAN minimum samples", 2, 20, 5)


if len(selected_features) < 2:
    st.warning("Select at least two numerical behaviour features in the sidebar.")
    st.stop()


features = df[selected_features].apply(
    pd.to_numeric,
    errors="coerce"
)

features = features.dropna(axis=1, how="all")

if features.shape[1] < 2:
    st.error("At least two usable numeric features are required.")
    st.stop()

features = features.fillna(features.median())
features = features.loc[:, features.nunique() > 1]

if features.shape[1] < 2:
    st.error("At least two non-constant features are required.")
    st.stop()

scaler = StandardScaler()
scaled = scaler.fit_transform(features)


# =========================================================
# MODEL EXECUTION
# =========================================================

kmeans_labels = None
dbscan_labels = None
kmeans_score = None
dbscan_score = None

if algorithm in ["K-Means", "Compare Both"]:
    kmeans_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans_labels = kmeans_model.fit_predict(scaled)
    kmeans_score = calculate_silhouette(scaled, kmeans_labels)

if algorithm in ["DBSCAN", "Compare Both"]:
    dbscan_model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )
    dbscan_labels = dbscan_model.fit_predict(scaled)
    dbscan_score = calculate_silhouette(scaled, dbscan_labels)


if algorithm == "DBSCAN":
    active_labels = dbscan_labels
    active_score = dbscan_score
    active_algorithm = "DBSCAN"
else:
    active_labels = kmeans_labels
    active_score = kmeans_score
    active_algorithm = "K-Means"


result = df.copy()
result["Cluster"] = active_labels

visual_df, pca_variance = create_pca(
    scaled,
    active_labels,
    df.index
)

profile_data = features.copy()
profile_data["Cluster"] = active_labels
profile = profile_data.groupby("Cluster").mean().round(2)

distribution = (
    pd.Series(active_labels)
    .value_counts()
    .sort_index()
)


# =========================================================
# PAGE 01: OVERVIEW
# =========================================================

if page == "01  Overview":

    st.markdown("""
    <div class="section-kicker">01 / SYSTEM OVERVIEW</div>
    <div class="section-heading">Your dataset, at a glance.</div>
    <div class="section-copy">
        A quick look at the scale and structure of your behavioural data.
    </div>
    """, unsafe_allow_html=True)

    a, b, c, d = st.columns(4)

    a.metric("TOTAL RECORDS", f"{len(df):,}")
    b.metric("FEATURES USED", len(features.columns))
    c.metric("SEGMENTS FOUND", cluster_count(active_labels))
    d.metric("NOISE POINTS", int(np.sum(active_labels == -1)))

    st.write("")

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown("### Dataset preview")
        st.dataframe(
            df.head(12),
            use_container_width=True,
            hide_index=True
        )

    with right:
        st.markdown("### Data quality")

        quality = pd.DataFrame({
            "Missing": df.isna().sum(),
            "Unique": df.nunique(),
            "Type": df.dtypes.astype(str)
        })

        st.dataframe(
            quality,
            use_container_width=True
        )

    st.divider()

    st.markdown("### Behaviour feature distribution")

    chosen_feature = st.selectbox(
        "Explore a feature",
        features.columns.tolist()
    )

    histogram = px.histogram(
        df,
        x=chosen_feature,
        nbins=30,
        template="plotly_dark",
        title=f"Distribution of {chosen_feature}"
    )

    histogram.update_traces(
        marker_color="#c5f36a",
        marker_line_width=0
    )

    histogram.update_layout(
        height=400,
        paper_bgcolor="#101310",
        plot_bgcolor="#191e19"
    )

    st.plotly_chart(histogram, use_container_width=True)


# =========================================================
# PAGE 02: CLUSTER STUDIO
# =========================================================

elif page == "02  Cluster Studio":

    st.markdown("""
    <div class="section-kicker">02 / MODEL WORKSPACE</div>
    <div class="section-heading">Cluster Studio.</div>
    <div class="section-copy">
        Investigate how your selected algorithm separates behavioural patterns.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"**Active algorithm:** `{active_algorithm}`"
    )

    if active_score is not None:
        st.metric("Silhouette score", f"{active_score:.3f}")
    else:
        st.info("Silhouette score is unavailable for this result.")

    fig = px.scatter(
        visual_df,
        x="PC 1",
        y="PC 2",
        color="Cluster",
        hover_data=["User ID"],
        template="plotly_dark",
        title="Behavioural map / PCA projection",
        color_discrete_sequence=[
            "#c5f36a",
            "#69c8a4",
            "#a68af2",
            "#f0a56b",
            "#68b8db",
            "#ed7e9a"
        ]
    )

    fig.update_traces(
        marker=dict(size=9, opacity=0.8)
    )

    fig.update_layout(
        height=600,
        paper_bgcolor="#101310",
        plot_bgcolor="#191e19",
        font_color="#edf1e9"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"The two PCA components explain {pca_variance:.1%} "
        "of the scaled feature variance."
    )

    if algorithm == "Compare Both":
        st.divider()
        st.markdown("### Algorithm comparison")

        comparison = pd.DataFrame({
            "Algorithm": ["K-Means", "DBSCAN"],
            "Clusters": [
                cluster_count(kmeans_labels),
                cluster_count(dbscan_labels)
            ],
            "Noise points": [
                int(np.sum(kmeans_labels == -1)),
                int(np.sum(dbscan_labels == -1))
            ],
            "Silhouette score": [
                kmeans_score,
                dbscan_score
            ]
        })

        st.dataframe(comparison, use_container_width=True)


# =========================================================
# PAGE 03: SEGMENT EXPLORER
# =========================================================

elif page == "03  Segment Explorer":

    st.markdown("""
    <div class="section-kicker">03 / BEHAVIOURAL RESEARCH</div>
    <div class="section-heading">Explore the segments.</div>
    <div class="section-copy">
        Examine what makes each discovered group different.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Segment distribution")

    distribution_df = distribution.rename_axis(
        "Cluster"
    ).reset_index(name="Users")

    distribution_df["Cluster"] = distribution_df["Cluster"].apply(
        lambda x: "Noise" if x == -1 else f"Group {x}"
    )

    bar = px.bar(
        distribution_df,
        x="Cluster",
        y="Users",
        text="Users",
        template="plotly_dark",
        title="Number of users in each segment"
    )

    bar.update_traces(
        marker_color="#c5f36a",
        textposition="outside"
    )

    bar.update_layout(
        height=400,
        paper_bgcolor="#101310",
        plot_bgcolor="#191e19"
    )

    st.plotly_chart(bar, use_container_width=True)

    st.divider()

    st.markdown("### Behaviour profile matrix")

    profile_display = profile.copy()
    profile_display.index = [
        "Noise" if x == -1 else f"Group {x}"
        for x in profile_display.index
    ]

    st.dataframe(
        profile_display,
        use_container_width=True
    )

    heatmap = px.imshow(
        profile_display,
        aspect="auto",
        color_continuous_scale=[
            "#202720",
            "#64843b",
            "#c5f36a"
        ],
        title="Average feature values by segment"
    )

    heatmap.update_layout(
        height=450,
        paper_bgcolor="#101310"
    )

    st.plotly_chart(heatmap, use_container_width=True)

    st.divider()

    st.markdown("### Inspect an individual segment")

    available_clusters = sorted(np.unique(active_labels).tolist())

    selected_cluster = st.selectbox(
        "Choose segment",
        available_clusters,
        format_func=lambda x: "Noise / Outliers" if x == -1 else f"Group {x}"
    )

    segment = result.loc[
        active_labels == selected_cluster
    ]

    st.metric("Users in selected segment", len(segment))

    st.dataframe(
        segment,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PAGE 04: DATA EXPORT
# =========================================================

elif page == "04  Data Export":

    st.markdown("""
    <div class="section-kicker">04 / OUTPUT & REPORTING</div>
    <div class="section-heading">Your analysis, ready to take away.</div>
    <div class="section-copy">
        Export the model output and segment profiles for further analysis.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Export summary")

    a, b, c = st.columns(3)

    a.metric("Records processed", f"{len(result):,}")
    b.metric("Segments discovered", cluster_count(active_labels))
    c.metric("Features analysed", len(features.columns))

    st.divider()

    st.markdown("### 01 — Clustered dataset")
    st.write(
        "Includes the original records and their assigned cluster labels."
    )

    st.download_button(
        "Download clustered CSV",
        result.to_csv(index=False).encode("utf-8"),
        file_name="customer_behaviour_clusters.csv",
        mime="text/csv"
    )

    st.divider()

    st.markdown("### 02 — Segment profiles")
    st.write(
        "Contains the average selected feature values for each cluster."
    )

    st.download_button(
        "Download segment profiles",
        profile.to_csv().encode("utf-8"),
        file_name="customer_segment_profiles.csv",
        mime="text/csv"
    )

    st.divider()

    st.markdown("### 03 — PCA coordinates")

    st.write(
        "A two-dimensional representation of the scaled data and cluster labels."
    )

    st.download_button(
        "Download PCA coordinates",
        visual_df.to_csv(index=False).encode("utf-8"),
        file_name="customer_pca_coordinates.csv",
        mime="text/csv"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    BEHAVIOUR LAB &nbsp; / &nbsp; DAY 04 OF 60
    <br><br>
    UNSUPERVISED LEARNING • CUSTOMER INTELLIGENCE • PYTHON
</div>
""", unsafe_allow_html=True)
