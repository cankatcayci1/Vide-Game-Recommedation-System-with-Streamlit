import pandas as pd
import numpy as np
import streamlit as st
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
def getFileAbsolutePath(filename):
    if '__file__' in globals():
        nb_dir = os.path.dirname(os.path.abspath(__file__)) 
    else:
        nb_dir = os.getcwd() 

    data_dir = os.path.join(nb_dir, 'Dataset')
    data_file = os.path.join(data_dir, filename)

    return data_file

data_file = getFileAbsolutePath('Video Games Sales.csv')
video_games_df = pd.read_csv(data_file)

video_games_filtered_df = video_games_df[['Name', 'Platform', 'Genre', 'Critic_Score', 'User_Score', 'Rating']]
video_games_filtered_df.dropna(subset=['Name', 'Genre', 'Rating'], axis=0, inplace=True)
video_games_filtered_df = video_games_filtered_df.reset_index(drop=True)

video_games_filtered_df['User_Score'] = np.where(video_games_filtered_df['User_Score'] == 'tbd', 
                                                 np.nan, 
                                                 video_games_filtered_df['User_Score']).astype(float)

video_game_grpby_genre = video_games_filtered_df[['Genre', 'Critic_Score', 'User_Score']].groupby('Genre', as_index=False)
video_game_score_mean = video_game_grpby_genre.agg(Ave_Critic_Score = ('Critic_Score', 'mean'), Ave_User_Score = ('User_Score', 'mean'))

video_games_filtered_df = video_games_filtered_df.merge(video_game_score_mean, on='Genre')
video_games_filtered_df['Critic_Score_Imputed'] = np.where(video_games_filtered_df['Critic_Score'].isna(), 
                                                           video_games_filtered_df['Ave_Critic_Score'], 
                                                           video_games_filtered_df['Critic_Score'])

video_games_filtered_df['User_Score_Imputed'] = np.where(video_games_filtered_df['User_Score'].isna(), 
                                                         video_games_filtered_df['Ave_User_Score'], 
                                                         video_games_filtered_df['User_Score'])
video_games_final_df = video_games_filtered_df.drop(columns=['User_Score', 'Critic_Score', 'Ave_Critic_Score', 'Ave_User_Score'], axis=1)
video_games_final_df = video_games_final_df.reset_index(drop=True)
video_games_final_df = video_games_final_df.rename(columns={'Critic_Score_Imputed':'Critic_Score', 'User_Score_Imputed':'User_Score'})

categorical_columns = [name for name in video_games_final_df.columns if video_games_final_df[name].dtype=='O']
categorical_columns = categorical_columns[1:]
video_games_df_dummy = pd.get_dummies(data=video_games_final_df, columns=categorical_columns)

features = video_games_df_dummy.drop(columns=['Name'], axis=1)
scale = StandardScaler()
scaled_features = scale.fit_transform(features)
scaled_features = pd.DataFrame(scaled_features, columns=features.columns)

model = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute').fit(scaled_features)
vg_distances, vg_indices = model.kneighbors(scaled_features)

game_names = video_games_df_dummy['Name'].drop_duplicates()
game_names = game_names.reset_index(drop=True)
vectorizer = TfidfVectorizer(use_idf=True).fit(game_names)
game_title_vectors = vectorizer.transform(game_names)

def VideoGameTitleRecommender(video_game_name):
    query_vector = vectorizer.transform([video_game_name])
    similarity_scores = cosine_similarity(query_vector, game_title_vectors)
    closest_match_index = similarity_scores.argmax()
    closest_match_game_name = game_names[closest_match_index]
    return closest_match_game_name

def VideoGameRecommender(video_game_name, video_game_platform='Any'):
    default_platform = 'Any'
    if video_game_platform != default_platform:
        video_game_idx = video_games_final_df.query("Name == @video_game_name & Platform == @video_game_platform").index
        if video_game_idx.empty:
            video_game_idx = video_games_final_df.query("Name == @video_game_name").index
            if not video_game_idx.empty:
                st.write(f"Note: Recommendations will be based on the title of the game as it is not available on the specified platform.\n")
                video_game_platform = default_platform
    else:
        video_game_idx = video_games_final_df.query("Name == @video_game_name").index  
    
    if video_game_idx.empty:
        closest_match_game_name = VideoGameTitleRecommender(video_game_name)
        st.write(f"'{video_game_name}' doesn't exist in the records.\n")
        st.write(f"You may want to try '{closest_match_game_name}', which is the closest match to the input.")
    else:
        game_details = video_games_final_df.iloc[video_game_idx[0]][['Name', 'Platform', 'Genre', 'Critic_Score', 'User_Score', 'Rating']]
        st.write(f"Details of the entered game '{video_game_name}':")
        st.write(game_details.to_frame().transpose())
        
        if video_game_platform == default_platform:
            vg_combined_dist_idx_df = pd.DataFrame()
            for idx in video_game_idx:
                vg_dist_idx_df = pd.concat([pd.DataFrame(vg_indices[idx][1:]), pd.DataFrame(vg_distances[idx][1:])], axis=1)
                vg_combined_dist_idx_df = pd.concat([vg_combined_dist_idx_df, vg_dist_idx_df])
            vg_combined_dist_idx_df.columns = ['Index', 'Distance']
            vg_combined_dist_idx_df = vg_combined_dist_idx_df.reset_index(drop=True)
            vg_combined_dist_idx_df = vg_combined_dist_idx_df.sort_values(by='Distance', ascending=True)
            video_game_list = video_games_final_df.iloc[vg_combined_dist_idx_df['Index']]
            video_game_list = video_game_list.drop_duplicates(subset=['Name'], keep='first')
            video_game_list = video_game_list.head(10)
            recommended_distances = np.array(vg_combined_dist_idx_df['Distance'].head(10))
        else:
            recommended_idx = vg_indices[video_game_idx[0]][1:]
            video_game_list = video_games_final_df.iloc[recommended_idx]
            recommended_distances = np.array(vg_distances[video_game_idx[0]][1:])
        
        st.write(f"Top 10 Recommended Video Games for '{video_game_name}'Platform: '{video_game_platform}'")
        video_game_list = video_game_list.reset_index(drop=True)
        recommended_video_game_list = pd.concat([video_game_list, 
                                                 pd.DataFrame(recommended_distances, columns=['Similarity_Distance'])], axis=1)
        st.write(recommended_video_game_list[['Name', 'Platform', 'Genre', 'Rating', 'Critic_Score', 'User_Score', 'Similarity_Distance']])

def main():
    st.title("Video Game Recommender System")
    game_name = st.text_input("Enter the name of the game:")
    platform_list = ['Any'] + list(video_games_df['Platform'].unique())
    game_platform = st.selectbox("Select the platform:", platform_list)
    
    if st.button("Get Recommendations"):
        if game_name:
            VideoGameRecommender(game_name, game_platform)
        else:
            st.write("Please enter a game name.")

if __name__ == "__main__":
    main()
