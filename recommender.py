import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class SongRecommender:
    def __init__(self):
        # Initialize with some sample song data
        self.song_data = pd.DataFrame(
            [
                {
                    "title": "Blinding Lights",
                    "artist": "The Weeknd",
                    "genre": "pop",
                    "tempo": 120,
                    "danceability": 0.8,
                    "energy": 0.7,
                    "mood": "happy",
                    "acousticness": 0.1,
                },
                {
                    "title": "Save Your Tears",
                    "artist": "The Weeknd",
                    "genre": "pop",
                    "tempo": 118,
                    "danceability": 0.75,
                    "energy": 0.65,
                    "mood": "melancholic",
                    "acousticness": 0.2,
                },
                {
                    "title": "Levitating",
                    "artist": "Dua Lipa",
                    "genre": "pop",
                    "tempo": 115,
                    "danceability": 0.85,
                    "energy": 0.8,
                    "mood": "happy",
                    "acousticness": 0.05,
                },
                {
                    "title": "Smooth",
                    "artist": "Santana",
                    "genre": "rock",
                    "tempo": 117,
                    "danceability": 0.7,
                    "energy": 0.75,
                    "mood": "happy",
                    "acousticness": 0.1,
                },
                {
                    "title": "Bohemian Rhapsody",
                    "artist": "Queen",
                    "genre": "rock",
                    "tempo": 72,
                    "danceability": 0.4,
                    "energy": 0.6,
                    "mood": "dramatic",
                    "acousticness": 0.4,
                },
                {
                    "title": "Shape of You",
                    "artist": "Ed Sheeran",
                    "genre": "pop",
                    "tempo": 96,
                    "danceability": 0.8,
                    "energy": 0.65,
                    "mood": "romantic",
                    "acousticness": 0.1,
                },
                {
                    "title": "Uptown Funk",
                    "artist": "Mark Ronson",
                    "genre": "funk",
                    "tempo": 115,
                    "danceability": 0.9,
                    "energy": 0.85,
                    "mood": "happy",
                    "acousticness": 0.3,
                },
                {
                    "title": "Billie Jean",
                    "artist": "Michael Jackson",
                    "genre": "pop",
                    "tempo": 117,
                    "danceability": 0.8,
                    "energy": 0.7,
                    "mood": "groovy",
                    "acousticness": 0.2,
                },
                {
                    "title": "Stay",
                    "artist": "The Kid LAROI",
                    "genre": "pop",
                    "tempo": 170,
                    "danceability": 0.6,
                    "energy": 0.5,
                    "mood": "melancholic",
                    "acousticness": 0.1,
                },
                {
                    "title": "Bad Guy",
                    "artist": "Billie Eilish",
                    "genre": "electropop",
                    "tempo": 135,
                    "danceability": 0.7,
                    "energy": 0.5,
                    "mood": "dark",
                    "acousticness": 0.2,
                },
            ]
        )

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
        print("\nEnter song titles you like (comma separated), or 'quit' to exit:")
        user_input = input().strip()

        if user_input.lower() == "quit":
            break

        input_songs = [s.strip() for s in user_input.split(",")]
        recommendations = recommender.recommend(input_songs)

        if isinstance(recommendations, str):
            print(recommendations)
        else:
            print("\nRecommended songs:")
            print(recommendations.to_string(index=False))
