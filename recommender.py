import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class SongRecommender:
    def __init__(self):
        # load song data from csv into a dataframe
        self.song_data = pd.read_csv("data/spotify_cleaned.csv")

        # Add unique ID column if it doesn't exist
        if "song_id" not in self.song_data.columns:
            self.song_data["song_id"] = [
                f"song_{i}" for i in range(len(self.song_data))
            ]

        # Prepare the data for similarity calculation
        self.prepare_data()

    def prepare_data(self):
        """Prepare the data for similarity calculations"""
        data = self.song_data.copy()

        # One-hot encode categorical features
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
        new_id = f"song_{len(self.song_data)}"

        new_song = pd.DataFrame(
            [
                {
                    "song_id": new_id,
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
        return new_id

    def recommend(self, song_ids, n_recommendations=5):
        """Recommend similar songs based on input song IDs"""
        # Find indices of input songs
        indices = []
        for song_id in song_ids:
            match = self.song_data[self.song_data["song_id"] == song_id]
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
            ["song_id", "title", "artist", "genre", "tempo", "danceability", "mood"]
        ]

    def get_song_by_id(self, song_id):
        """Get song details by ID"""
        return self.song_data[self.song_data["song_id"] == song_id]

    def search_songs(self, query):
        """Search songs by title or artist"""
        return self.song_data[
            self.song_data["title"].str.contains(query, case=False)
            | self.song_data["artist"].str.contains(query, case=False)
        ]


# Example usage
if __name__ == "__main__":
    recommender = SongRecommender()

    print("Welcome to the Song Recommender System!")
    print("Available songs in database:")
    print(
        recommender.song_data[["song_id", "title", "artist", "genre"]].to_string(
            index=False
        )
    )

    while True:
        print("\nEnter a song title or artist to search for:")
        user_input = input().strip()
        matches = recommender.search_songs(user_input)
        if matches.empty:
            print("No matching song found.")
            continue
        print("Matching songs:")
        print(matches[["song_id", "title", "artist", "genre"]].to_string(index=False))
        print("\nEnter the song ID of the song you want to select:")
        selected_id = input().strip()
        if selected_id not in matches["song_id"].values:
            print("Invalid song ID.")
            continue
        print(f"Searching for recommendations...")

        recommendations = recommender.recommend([selected_id])

        if isinstance(recommendations, str):
            print(recommendations)
        else:
            print("\nRecommended songs:")
            print(recommendations.to_string(index=False))
