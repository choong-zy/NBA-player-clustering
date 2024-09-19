import os
import pickle
from sklearn.decomposition import PCA
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category20_12  # Import Category20_12 instead of Category20
from bokeh.transform import factor_cmap
import plotly.express as px
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering, Birch, MeanShift, SpectralClustering
from fcmeans import FCM
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from math import pi

# Load data and models
df = pd.read_csv('final_df.csv')
model1 = joblib.load("model/HCmodel.pkl")
model2 = joblib.load("model/SCmodel.pkl")
model3 = joblib.load("model/GMMmodel.pkl")
model4 = joblib.load("model/MSmodel.pkl")

st.set_page_config(layout='wide')

# Define features
features = ['Player', 'Pos', 'PER', 'BPM', 'WS/48', 'VORP', 'TS%', 'FG%', 'eFG%', '3P%', '2P%', 'FT%', 'PPG', 'APG', 'RPG', 'SPG', 'BPG', 'TPG', 'FPG']
df = df[features]

class ClusterSimilarityMatrix():

    def __init__(self) -> None:
        self._is_fitted = False

    def fit(self, y_clusters):
        if not self._is_fitted:
            self._is_fitted = True
            self.similarity = self.to_binary_matrix(y_clusters)
            return self

        self.similarity += self.to_binary_matrix(y_clusters)

    def to_binary_matrix(self, y_clusters):
        y_reshaped = np.expand_dims(y_clusters, axis=-1)
        return (cdist(y_reshaped, y_reshaped, 'cityblock')==0).astype(int)

class EnsembleCustering():
    def __init__(self, base_estimators, aggregator, distances=False):
        self.base_estimators = base_estimators
        self.aggregator = aggregator
        self.distances = distances

    def fit(self, X):
        X_ = X.copy()

        clt_sim_matrix = ClusterSimilarityMatrix()
        for model in self.base_estimators:
            clt_sim_matrix.fit(model.fit_predict(X=X_))

        sim_matrix = clt_sim_matrix.similarity
        self.cluster_matrix = sim_matrix/sim_matrix.diagonal()

        if self.distances:
            self.cluster_matrix = np.abs(np.log(self.cluster_matrix + 1e-8)) # Avoid log(0)

    def fit_predict(self, X):
        self.fit(X)
        y = self.aggregator.fit_predict(self.cluster_matrix)
        return y

def preprocess_data(data):
    # Declare perf_features at the start
    perf_features = ['PER', 'BPM', 'WS/48', 'VORP', 'TS%', 'FG%', 'eFG%', '3P%', '2P%', 'FT%', 'PPG', 'APG', 'RPG', 'SPG', 'BPG', 'TPG', 'FPG']

    scaler = StandardScaler()

    # Scale numeric features
    data[perf_features] = scaler.fit_transform(data[perf_features])
    X = data[perf_features]
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='pca', random_state = 21).fit_transform(X)
    tsne_df = pd.DataFrame({'component_1': X_embedded[:,0], 'component_2': X_embedded[:,1]})

    # Add 'Pos' and 'Player' columns to tsne_df
    tsne_df['Pos'] = data['Pos']
    tsne_df['Player'] = data['Player']

    return tsne_df

# Home page
def home():
    st.title("NBA Player Performance Clustering")
    st.write("Welcome to the NBA Player Performance Clustering app. This tool allows you to explore clusters of NBA players based on their performance metrics.")
    
    # Set background image
    background_image = 'image2.png'  # Replace with your image path
    with open(background_image, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title and introduction
    st.title("NBA Player Performance Clustering")
    st.write("""
    Welcome to the NBA Player Performance Clustering app! This tool uses machine learning 
    to group NBA players based on their performance metrics. Explore player clusters, 
    analyze performance trends, and predict player positions using advanced statistics.
    """)

    # Sample visualization
    st.subheader("Player Clustering Visualization")

    # Load saved ensemble clustering model, final t-SNE DataFrame, and cluster labels
    model_path = os.path.join(os.getcwd(), 'ensemble_clustering_model.pkl')
    tsne_df_path = os.path.join(os.getcwd(), 'final_tsne_df.pkl')
    labels_path = os.path.join(os.getcwd(), 'ensemble_labels.pkl')

    with open("model/ensemble_clustering_model.pkl", 'rb') as f:
        ensemble_clustering_model = pickle.load(f)

    with open("model/final_tsne_df.pkl", 'rb') as f:
        tsne_df = pickle.load(f)

    with open("model/ensemble_labels.pkl", 'rb') as f:
        y_ensemble = pickle.load(f)

    # Add 'Player_Name' and 'Position' columns to tsne_df
    tsne_df['Player_Name'] = df['Player']
    tsne_df['Position'] = df['Pos']

    # Add 'Cluster_Ensemble' column to tsne_df
    tsne_df['Cluster_Ensemble'] = y_ensemble

    # Ensure all required columns are present
    required_columns = ['Player_Name', 'Position', 'Cluster_Ensemble', 'component_1', 'component_2']
    for col in required_columns:
        if col not in tsne_df.columns:
            raise ValueError(f"Required column '{col}' is missing from tsne_df")

    # Data source for Bokeh
    def filter_data(cluster_value, selected_position):
        """Filter the data based on the cluster value and selected position."""
        if cluster_value == 3:  # Show all clusters
            mask = tsne_df['Position'] == selected_position if selected_position != "All" else pd.Series([True] * len(tsne_df))
        else:
            mask = (y_ensemble == cluster_value) & (tsne_df['Position'] == selected_position if selected_position != "All" else pd.Series([True] * len(tsne_df)))
        return tsne_df[mask]

    # Create Bokeh plot
    hover = HoverTool(tooltips=[("Player", "@Player_Name{safe}"), ("Position", "@Position{safe}")], point_policy="follow_mouse")

    cluster_mapping = {
        0: "Bench & Developing Players",
        1: "Defensive & Role Players",
        2: "All-Star Players"
    }

    def create_plot(data, labels):
        # Map numeric labels to cluster names
        cluster_names = [cluster_mapping[label] for label in labels]
        
        source = ColumnDataSource(data=dict(
            x=data['component_1'],
            y=data['component_2'],
            Player_Name=data['Player_Name'],
            Position=data['Position'],
            cluster=cluster_names
        ))

        # Use a color palette with distinct colors for each cluster
        palette = Category20_12  # Adjusted palette for 12 distinct colors

        # Use factor_cmap to map cluster names to colors
        mapper = factor_cmap('cluster', palette=palette, factors=list(cluster_mapping.values()))

        plot = figure(width=700, height=500, tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'tap'],
                    title="Clustering of NBA Players with t-SNE and Ensemble Clustering", toolbar_location="above")

        plot.scatter('x', 'y', size=5, source=source, fill_color=mapper, line_alpha=0.3, line_color="black", legend_group='cluster')

        # Improve legend and place it outside the plot
        plot.add_layout(plot.legend[0], 'right')
        plot.legend.title = "Cluster Types"
        plot.legend.click_policy = "hide"
        plot.legend.label_text_font_size = "8pt"
        plot.legend.spacing = 1
        plot.legend.glyph_height = 15

        return plot
    
    # Add widgets for filtering
    slider = st.slider("Cluster #", 0, 3, 3, help="Slide to filter clusters, 0-2 for individual clusters, 3 for all clusters")
    selected_position = st.selectbox("Select Position:", ["All", "C", "PF", "SF", "PG", "SG"], help="Select player position to filter")

    # Display a title for the selected cluster in smaller font using HTML
    if 0 <= slider <= 2:
        st.markdown(f"<h3>Cluster {slider} Players</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3>All Players</h3>", unsafe_allow_html=True)

    # Filter data based on the selected cluster and search term
    filtered_data = filter_data(slider, selected_position)

    # Create the updated plot
    plot = create_plot(filtered_data, filtered_data['Cluster_Ensemble'])

    # Display the plot in Streamlit
    st.bokeh_chart(plot)

# Dashboard page
def dashboard():
    st.title("NBA Player Clustering Dashboard")

    # Load data
    df = pd.read_csv('final_df.csv')
    
    # Separate numeric and non-numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_columns]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ... rest of the dashboard function ...
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Data Exploration", "Data Analysis", "t-SNE Visualization", "Comparative Analysis"])

    # Data Exploration
    with tab1:
        st.write("Data Preview:", df.head())
        column_to_filter = st.selectbox("Select a column to filter:", df.columns)
        
        if pd.api.types.is_numeric_dtype(df[column_to_filter]):
            filter_value = st.slider(f"Select a value for {column_to_filter}:", 
                                    min_value=float(df[column_to_filter].min()), 
                                    max_value=float(df[column_to_filter].max()))
            filtered_data = df[df[column_to_filter] == filter_value]
        else:
            filter_value = st.selectbox(f"Select a value for {column_to_filter}:", df[column_to_filter].unique())
            filtered_data = df[df[column_to_filter] == filter_value]
        
        st.write("Filtered Data", filtered_data)

    # Interactive Charts
    with tab2:
        st.header("Data Analysis")

        # Histograms
        st.subheader("Histograms of Numeric Features")
        fig, ax = plt.subplots(figsize=(14, 18))
        df[numeric_columns].hist(bins=20, figsize=(14, 18), ax=ax)
        st.pyplot(fig)

        # Correlation Matrix
        st.subheader("Correlation Matrix")
        numeric_cols = ['PER', 'BPM', 'WS/48', 'VORP', 'TS%', 'FG%', 'eFG%', '3P%', '2P%', 'FT%', 'PPG', 'APG', 'RPG', 'SPG', 'BPG', 'TPG', 'FPG']
        correlation_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(20, 16))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        plt.title('Correlation Matrix of Statistics')
        st.pyplot(fig)

        # Boxplots
        st.subheader("Boxplots of Numeric Features")
        plots_per_row = 5
        fig, axes = plt.subplots(nrows=len(numeric_cols)//plots_per_row + 1, ncols=plots_per_row, figsize=(20, len(numeric_cols)//plots_per_row * 5))
        axes = axes.flatten()
        for i, column in enumerate(numeric_cols):
            sns.boxplot(x=df[column], ax=axes[i])
            axes[i].set_title(f'Box plot of {column}')
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        st.pyplot(fig)

        # Position Distribution
        st.subheader("Position Distribution")
        pos_count = df['Pos'].value_counts()
        fig, ax = plt.subplots(figsize=(3, 3))
        pos_count.plot.pie(y='Pos', autopct='%1.1f%%', startangle=90, ax=ax)
        st.pyplot(fig)

        # Position-Based Analysis
        st.subheader("Average Statistics by Position")
        numeric_cols = ['3P%', '2P%', 'FT%', 'PPG', 'APG', 'RPG', 'SPG', 'BPG', 'TPG', 'FPG']
        data_filtered = df[['Pos'] + numeric_cols]
        data_means = data_filtered.groupby('Pos').mean().reset_index()
        fig, ax = plt.subplots(figsize=(14, 8))
        data_means.set_index('Pos').plot(kind='bar', ax=ax)
        plt.title('Average Statistics by Position')
        plt.xlabel('Position')
        plt.ylabel('Average Value')
        plt.legend(title='Statistic')
        st.pyplot(fig)

        # Offensive vs Defensive Stats
        st.subheader("Offensive vs Defensive Box Plus/Minus")
        bpm_sorted = df.sort_values(by='PER', ascending=False)
        top_bpm = bpm_sorted[:194]
        filtered_top_bpm = top_bpm.loc[:, ['Player', 'OBPM', 'DBPM', 'BPM']]
        fig, ax = plt.subplots(figsize=(16, 16))
        plt.scatter(filtered_top_bpm['OBPM'], filtered_top_bpm['DBPM'])
        for i in range(30):
            name = filtered_top_bpm.iloc[i]['Player']
            plt.annotate(name, xy=(filtered_top_bpm.iloc[i]['OBPM'], filtered_top_bpm.iloc[i]['DBPM']),
                         xycoords="data", textcoords='offset points', xytext=(3, 3), fontsize='small')
        plt.xlabel("Offensive Box Plus/Minus", size="xx-large")
        plt.ylabel("Defensive Box Plus/Minus", size="xx-large")
        plt.title("Comparison of OBPM and DBPM for all players along with names for top 30 players in BPM", size="xx-large")
        st.pyplot(fig)

        def plot_performance_radar(player_name, data, categories):
            def normalize(series):
                """Normalize series to a 0-10 scale."""
                min_val = series.min()
                max_val = series.max()
                return 10 * (series - min_val) / (max_val - min_val)

            player_stats = data[data['Player'] == player_name]
            if not player_stats.empty:
                # Normalize all relevant statistics
                normalized_data = data[categories].apply(normalize)

                player_normalized_stats = normalized_data[data['Player'] == player_name].iloc[0]
                # Prepare radar chart data
                N = len(categories)
                angles = [n / float(N) * 2 * pi for n in range(N)]
                player_stats_values = np.append(player_normalized_stats.values, player_normalized_stats.values[0])
                angles += angles[:1]

                # Create figure and axis objects
                fig, ax = plt.subplots(figsize=(2, 2), subplot_kw=dict(polar=True))
                ax.plot(angles, player_stats_values, linewidth=1, linestyle='solid')
                ax.fill(angles, player_stats_values, 'b', alpha=0.1)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories, color='grey', size=8)
                ax.set_title(f'Performance Radar for {player_name}', size=10)
                
                return fig
            else:
                print(f"{player_name} not found in the dataset.")
                return None

        # Radar Charts
        st.subheader("Player Performance Radar Chart")
        categories = ['PPG', 'APG', 'RPG', 'SPG', 'BPG']

        # Sort players by VORP and get top 50
        VORP_sorted = df.sort_values(by='VORP', ascending=False)
        top_players = VORP_sorted['Player'].head(50).tolist()

        # Create dropdown for player selection
        selected_player = st.selectbox("Select a player:", top_players)

        # Plot radar chart for selected player
        fig = plot_performance_radar(selected_player, df, categories)
        if fig:
            st.pyplot(fig)
            plt.close(fig)  # Close the figure to free up memory
        else:
            st.write("No data available for the selected player.")

    # t-SNE Visualization
    with tab3:
        # Prepare the data
        perf_features = ['PER', 'BPM', 'WS/48', 'VORP', 'TS%', 'FG%', 'eFG%', '3P%', '2P%', 'FT%', 'PPG', 'APG', 'RPG', 'SPG', 'BPG', 'TPG', 'FPG']
        X = df[perf_features]
            
            # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        tsne = TSNE(n_components=2, learning_rate='auto', init='pca', random_state = 21)
        tsne_result = tsne.fit_transform(X_scaled)
        tsne_df = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2'])
        tsne_df['Player'] = df['Player']
        tsne_df['Position'] = df['Pos']
        
        fig = px.scatter(tsne_df, x='TSNE1', y='TSNE2', color='Position', hover_data=['Player'],
                        title="t-SNE 2D Visualization of NBA Players")
        st.plotly_chart(fig)

    # Comparative Analysis
    with tab4:
        st.markdown("## Comparative Analysis of Models")
        model_options = ['GMM', 'Agglomerative Clustering', 'Spectral Clustering', 'Birch', 'Mean Shift', 'FCMeans', 'Ensemble']
        selected_models = st.multiselect("Select models to compare:", model_options)
        
        if selected_models:
            n_clusters = st.slider("Number of Clusters for Comparison", min_value=2, max_value=10, value=3)
            
            fig = make_subplots(rows=len(selected_models), cols=1, subplot_titles=selected_models)
            metrics = []
            
            # Prepare the data
            perf_features = ['PER', 'BPM', 'WS/48', 'VORP', 'TS%', 'FG%', 'eFG%', '3P%', '2P%', 'FT%', 'PPG', 'APG', 'RPG', 'SPG', 'BPG', 'TPG', 'FPG']
            X = df[perf_features]
            
            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform t-SNE
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=21)
            X_tsne = tsne.fit_transform(X_scaled)
            tsne_df = pd.DataFrame({'component_1': X_tsne[:,0], 'component_2': X_tsne[:,1]})
            
            for i, model_type in enumerate(selected_models, start=1):
                if model_type == 'GMM':
                    model = GaussianMixture(n_components=n_clusters, covariance_type='full', max_iter=1000,  tol=0.001)
                elif model_type == 'Agglomerative Clustering':
                    model = AgglomerativeClustering(n_clusters=n_clusters, affinity='chebyshev', linkage='average')
                elif model_type == 'Spectral Clustering':
                    model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', eigen_solver='arpack', n_neighbors=20, assign_labels='discretize')
                elif model_type == 'Birch':
                    model = Birch(n_clusters=n_clusters, threshold=1.0, branching_factor=150)
                elif model_type == 'Mean Shift':
                    model = MeanShift(bandwidth=12, cluster_all=True, max_iter=100, bin_seeding=True)
                elif model_type == 'FCMeans':
                    model = FCM(n_clusters=n_clusters, m=1.5, error=0.001, max_iter=100)
                else:  # Ensemble
                    HCmodel = AgglomerativeClustering(n_clusters=3, affinity='chebyshev', linkage='average')
                    GMMmodel = GaussianMixture(n_components=3, covariance_type='full', max_iter=1000, tol=0.001)
                    MSmodel = MeanShift(bandwidth=2, cluster_all=True, max_iter=100, bin_seeding=True)
                    SCmodel = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', eigen_solver='arpack', n_neighbors=20, assign_labels='discretize')
                    base_estimators = [HCmodel, SCmodel, GMMmodel, MSmodel]
                    aggregator_clt = GaussianMixture(n_components=3, covariance_type='full', max_iter=1000, tol=0.001)
                    model = EnsembleCustering(base_estimators=base_estimators, aggregator=aggregator_clt)

                predicted_labels = model.fit_predict(X_tsne)
                
                fig.add_trace(
                    go.Scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], mode='markers',
                            marker=dict(color=predicted_labels, colorscale='Viridis', showscale=False),
                            text=df['Player'], hoverinfo='text'),
                    row=i, col=1
                )
                
                silhouette_avg = silhouette_score(X_tsne, predicted_labels)
                davies_bouldin = davies_bouldin_score(X_tsne, predicted_labels)
                calinski_harabasz = calinski_harabasz_score(X_tsne, predicted_labels)
                metrics.append([model_type, silhouette_avg, davies_bouldin, calinski_harabasz])
            
            fig.update_layout(height=300*len(selected_models), title_text="Comparative Cluster Analysis")
            st.plotly_chart(fig)
            
            metrics_df = pd.DataFrame(metrics, columns=['Model', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz'])
            st.write("Clustering Metrics Comparison:")
            st.write(metrics_df)

    st.sidebar.info("This dashboard allows you to explore NBA player clustering using various techniques and visualizations.")

# Predict page
def predict():
    #st.markdown("<h1 style='text-align: center; color: red;'>NBA Player Position Clustering</h1>", unsafe_allow_html=True)
    #st.title("NBA :blue[Player Position Clustering]")
    st.title("NBA Player Position Clustering")

    # Input form from the user
    col1, col2, col3 = st.columns(3)

    # Get input from the user
    with col1:
        player = st.text_input("Player Name")
        pos = st.radio("Position", ["Center (C)", "Power Forward (PF)", "Small Forward (SF)", "Point Guard (PG)", "Shooting Guard (SG)"])
        per = st.number_input("Player Efficiency Rating (PER)")
        bpm = st.number_input("Box Plus/Minus (BPM)")
        ws = st.number_input("Win Shares Per 48 Minutes (WS/48)")
        vorp = st.number_input("Value over Replacement Player (VORP)")
        st.markdown('#')
        st.markdown('#')

    with col2:
        ts = st.number_input("True Shooting Percentage (TS%)")
        fg = st.number_input("Field Goal Percentage (FG%)")
        efg = st.number_input("Effective Field Goal Percentage (eFG%)")
        threePoint = st.number_input("3-Point Field Goal Percentage (3P%)")
        twoPoint = st.number_input("2-Point Field Goal Percentage (2P%)")
        ft = st.number_input("Free Throw Percentage (FT%)")
        st.markdown('#')

    with col3:
        ppg = st.number_input("Points Per Game (PPG)")
        apg = st.number_input("Assist Per Game (APG)")
        rpg = st.number_input("Rebound Per Game (RPG)")
        spg = st.number_input("Steal Per Game (SPG)")
        bpg = st.number_input("Block Per Game (BPG)")
        tpg = st.number_input("Turnover Per Game (TPG)")
        fpg = st.number_input("Foul Per Game (FPG)")
    
    # Create a dictionary with the input data
    input_data = {
        'Player': player,
        'Pos': pos,
        'PER': per,
        'BPM': bpm,
        'WS/48': ws,
        'VORP': vorp,
        'TS%': ts,
        'FG%': fg,
        'eFG%': efg,
        '3P%': threePoint,
        '2P%': twoPoint,
        'FT%': ft,
        'PPG': ppg,
        'APG': apg,
        'RPG': rpg,
        'SPG': spg,
        'BPG': bpg,
        'TPG': tpg,
        'FPG': fpg
    }

    input_df = pd.DataFrame([input_data])
    combined_df = pd.concat([df, input_df], ignore_index=True)
    combined_df = combined_df[features]

    with col2:
        if st.button("Cluster"):
            final_tsne_df = preprocess_data(combined_df)
            base_estimators = [
                model1,
                model2,
                model3,
                model4
            ]

            # Create the ensemble clustering object, using Spectral Clustering as the aggregator
            aggregator_clt = GaussianMixture(n_components=3, covariance_type='full', max_iter=1000, tol=0.001)
            ensemble_clustering = EnsembleCustering(base_estimators=base_estimators, aggregator=aggregator_clt)
            X_vectors = final_tsne_df[['component_1', 'component_2']].values
            final_tsne_df['Cluster'] = ensemble_clustering.fit_predict(X_vectors)

            new_row_cluster = final_tsne_df.iloc[[-1]]
            existing_data = final_tsne_df.iloc[:-1]


            new_row_cluster_id = new_row_cluster['Cluster'].values[0]
            result_info = pd.DataFrame([{"Name:": player, "Position:": pos, "Cluster:": new_row_cluster_id}])
            st.write("New player cluster information:",result_info)

            # Find all players in the same cluster as the new row
            same_cluster_players = combined_df[(final_tsne_df['Cluster'] == new_row_cluster_id) &
                                       (combined_df['Player'] != player)]

            # Show some players in the same cluster
            st.write(f"Players in the same cluster (Cluster {new_row_cluster_id}):")
            st.dataframe(same_cluster_players[['Player', 'Pos']].reset_index(drop=True))  # Show only Player and Position columns

            with col1:
                plt.figure(figsize=(10, 7))
                sns.scatterplot(x='component_1', y='component_2', hue='Cluster', data=existing_data, palette='Set1', legend='full', alpha=0.7)
                plt.scatter(new_row_cluster['component_1'], new_row_cluster['component_2'],
                            color='red', s=200, label='New Point', edgecolor='black')
                plt.title('Clustering')
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.legend(title='Cluster')
                st.pyplot(plt)
                plt.close()


def get_base64_encoded_audio(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")
    except Exception as e:
        st.error(f"Error reading audio file {file_path}: {str(e)}")
        return ""

def add_audio_elements():
    bgm_base64 = get_base64_encoded_audio("bgm.mp3")
    click_sound_base64 = get_base64_encoded_audio("click.mp3")
    
    st.markdown(f"""
    <audio id="bgm" autoplay loop style="display:none;">
        <source src="data:audio/mpeg;base64,{bgm_base64}" type="audio/mpeg">
    </audio>
    <audio id="clickSound" style="display:none">
        <source src="data:audio/mpeg;base64,{click_sound_base64}" type="audio/mpeg">
    </audio>
    <script>
        var bgm = document.getElementById("bgm");
        var clickSound = document.getElementById("clickSound");
        bgm.volume = 0.1;  // Adjust volume as needed

        // Attempt to play BGM on page load
        document.addEventListener("DOMContentLoaded", function() {{
            bgm.play().catch(function(error) {{
                console.log("Autoplay prevented. User interaction required.");
            }});
        }});

        // Play click sound on button clicks
        const buttons = window.parent.document.querySelectorAll("button");
        buttons.forEach(button => {{
            button.addEventListener("click", function() {{
                clickSound.play();
                if (bgm.paused) {{
                    bgm.play();
                }}
            }});
        }});
    </script>
    """, unsafe_allow_html=True)

# Main app logic
def main():
    st.sidebar.title("Navigation")

    # Add audio elements
    add_audio_elements()

    # Initialize page with a default value
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    
    # Custom CSS for the navigation buttons
    st.markdown(
        """
        <style>
        .nav-button {
            background-color: #f0f2f6;
            border: none;
            color: #262730;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
            width: 200px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Create clickable buttons for navigation
    if st.sidebar.button("Home", key="home", help="Go to Home page", type="primary", use_container_width=True, on_click=lambda: setattr(st.session_state, 'page', 'Home')):
        pass
    if st.sidebar.button("Dashboard", key="dashboard", help="Go to Dashboard page", type="primary", use_container_width=True, on_click=lambda: setattr(st.session_state, 'page', 'Dashboard')):
        pass
    if st.sidebar.button("Predict", key="predict", help="Go to Predict page", type="primary", use_container_width=True, on_click=lambda: setattr(st.session_state, 'page', 'Predict')):
        pass

    # Display the appropriate page based on the session state
    if st.session_state.page == "Home":
        home()
    elif st.session_state.page == "Dashboard":
        dashboard()
    elif st.session_state.page == "Predict":
        predict()

if __name__ == "__main__":
    main()