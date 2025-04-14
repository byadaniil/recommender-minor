import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class SongRecommender:
    def __init__(self):
        # load song data from csv into a dataframe
        self.song_data = pd.read_csv("data/spotify_cleaned.csv")

        # Prepare the data for similarity calculation
        self.prepare_data()

    def prepare_data(self):
        """Prepare the data for similarity calculations"""
        # One-hot encode categorical features
        data = self.song_data.copy()
        genres = pd.get_dummies(data["genre"], prefix="genre")
        moods = pd.get_dummies(data["mood"], prefix="mood")

        # Scale numerical features
        numerical_features = ["tempo", "danceability", "energy", "acousticness"]
        scaler = MinMaxScaler()
        scaled_numerical = scaler.fit_transform(data[numerical_features])
        scaled_numerical = pd.DataFrame(scaled_numerical, columns=numerical_features)

        # Combine all features
        self.features = pd.concat([genres, moods, scaled_numerical], axis=1)

        # Compute similarity matrix
        self.similarity_matrix = cosine_similarity(self.features)

    def add_song(
        self, title, artist, genre, tempo, danceability, energy, mood, acousticness
    ):
        """Add a new song to the database"""
        new_song = pd.DataFrame(
            [
                {
                    "title": title,
                    "artist": artist,
                    "genre": genre,
                    "tempo": tempo,
                    "danceability": danceability,
                    "energy": energy,
                    "mood": mood,
                    "acousticness": acousticness,
                }
            ]
        )

        self.song_data = pd.concat([self.song_data, new_song], ignore_index=True)
        self.prepare_data()

    def recommend(self, song_titles, n_recommendations=5):
        """Recommend similar songs based on input songs"""
        # Find indices of input songs
        indices = []
        for title in song_titles:
            match = self.song_data[
                self.song_data["title"].str.contains(title, case=False)
            ]
            if not match.empty:
                indices.append(match.index[0])

        if not indices:
            return "No matching songs found in database."

        # Calculate average similarity across input songs
        avg_similarity = np.mean(self.similarity_matrix[indices], axis=0)

        # Get top recommendations (excluding input songs)
        similar_indices = np.argsort(avg_similarity)[::-1]
        similar_indices = [i for i in similar_indices if i not in indices][
            :n_recommendations
        ]

        recommendations = self.song_data.iloc[similar_indices]
        return recommendations[
            ["title", "artist", "genre", "tempo", "danceability", "mood"]
        ]


# Example usage
if __name__ == "__main__":
    recommender = SongRecommender()

    print("Welcome to the Song Recommender System!")
    print("Available songs in database:")
    print(recommender.song_data[["title", "artist", "genre"]].to_string(index=False))

    while True:

        # let the user search for a song by title and then confirm selection
        print("\nEnter a song title to search for:")
        user_input = input().strip()
        match = recommender.song_data[
            recommender.song_data["title"].str.contains(user_input, case=False)
        ]
        if match.empty:
            print("No matching song found.")
            continue
        print("Matching songs:")
        print(match[["title", "artist", "genre"]].to_string(index=False))
        print("\nEnter the index of the song you want to select:")
        selected_index = int(input().strip())
        if selected_index < 0 or selected_index >= len(match):
            print("Invalid index.")
            continue
        selected_song = match.iloc[selected_index]
        input_songs = [selected_song["title"]]
        print(f"You selected: {selected_song['title']} by {selected_song['artist']}")
        print("Searching for recommendations...")

        recommendations = recommender.recommend(input_songs)

        if isinstance(recommendations, str):
            print(recommendations)
        else:
            print("\nRecommended songs:")
            print(recommendations.to_string(index=False))
