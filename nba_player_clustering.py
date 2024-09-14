import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture

# Load the pre-trained model
# df = pd.read_csv('/content/drive/MyDrive/ML/final_df.csv')
# model1 = joblib.load("/content/drive/MyDrive/ML/HCmodel.pkl")
# model2 = joblib.load("/content/drive/MyDrive/ML/SCmodel.pkl")
# model3 = joblib.load("/content/drive/MyDrive/ML/GMMmodel.pkl")
# model4 = joblib.load("/content/drive/MyDrive/ML/MSmodel.pkl")

df = pd.read_csv('final_df.csv')
model1 = joblib.load("model/HCmodel.pkl")
model2 = joblib.load("model/SCmodel.pkl")
model3 = joblib.load("model/GMMmodel.pkl")
model4 = joblib.load("model/MSmodel.pkl")
st.set_page_config(layout='wide')
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

def preprocess_input(data):
    # Declare perf_features at the start
    perf_features = ['PER', 'BPM', 'WS/48', 'VORP', 'TS%', 'FG%', 'eFG%', '3P%', '2P%', 'FT%', 'PPG', 'APG', 'RPG', 'SPG', 'BPG', 'TPG', 'FPG']

    scaler = StandardScaler()

    # Scale numeric features
    data[perf_features] = scaler.fit_transform(data[perf_features])
    X = data[perf_features]
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='pca', random_state = 21).fit_transform(X)
    tsne_df = pd.DataFrame({'component_1': X_embedded[:,0], 'component_2': X_embedded[:,1]})

    return tsne_df

# Streamlit app
def main():
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
            final_tsne_df = preprocess_input(combined_df)
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

if __name__ == "__main__":
    main()
