import streamlit as st
import pandas as pd
from recommender import SongRecommender


# Initialize the recommender system
@st.cache_resource
def load_recommender():
    return SongRecommender()


recommender = load_recommender()


def display_song(song):
    return f"{song['title']} by {song['artist']} ({song['genre']})"


# Streamlit app
def main():
    st.title("🎵 Song Recommender System")
    st.write("Discover songs similar to your favorites!")
    # Search for a song
    st.subheader("Search for a song")
    search_query = st.text_input("Enter a song title or artist:")

    # Initialize session state for selected songs
    if "selected_songs" not in st.session_state:
        st.session_state.selected_songs = {}  # {song_id: song_data}

    # Current search results
    current_matches = None

    if search_query:
        # Find matching songs
        matches = recommender.search_songs(search_query)

        if not matches.empty:
            current_matches = matches
            st.write("Matching songs found:")
            st.dataframe(matches, use_container_width=True)

            # Select songs from current search
            song_options = {
                row["song_id"]: display_song(row)
                for _, row in matches.iterrows()
                if row["song_id"] not in st.session_state.selected_songs
            }

            if song_options:
                songs_to_add = st.multiselect(
                    "Select songs to add to your selection:",
                    options=list(song_options.keys()),
                    format_func=lambda x: song_options[x],
                )

                if st.button("Add Selected Songs", key="add_songs_btn"):
                    for song_id in songs_to_add:
                        song = matches[matches["song_id"] == song_id].iloc[0]
                        st.session_state.selected_songs[song_id] = song
                    st.success(f"Added {len(songs_to_add)} song(s) to selection")
            else:
                st.info("All matching songs are already in your selection")
        else:
            st.warning("No matching songs found. Please try another search.")

    # Display and manage current selection
    st.subheader("Your Current Selection")

    if st.session_state.selected_songs:
        selected_songs_df = pd.DataFrame(st.session_state.selected_songs.values())
        st.table(
            selected_songs_df[
                [
                    "song_id",
                    "title",
                    "artist",
                    "genre",
                    "tempo",
                    "danceability",
                    "mood",
                ]
            ].set_index("song_id")
        )
        if st.button("Clear All Selections"):
            st.session_state.selected_songs = {}
            st.experimental_rerun()
        # Get recommendations
        if st.button("Get Recommendations Based on Selection", key="recommend_btn"):
            with st.spinner("Finding similar songs..."):
                recommendations = recommender.recommend(
                    list(st.session_state.selected_songs.keys()), n_recommendations=10
                )

            if isinstance(recommendations, str):
                st.warning(recommendations)
            else:
                st.subheader("Recommended songs:")
                st.table(
                    recommendations.set_index("song_id")[
                        ["title", "artist", "genre", "tempo", "danceability", "mood"]
                    ]
                )

    else:
        st.info(
            "No songs selected yet. Search for songs above to add to your selection."
        )

    with st.expander("Add New Song to Database"):
        with st.form("add_song_form"):
            st.write("Add a new song to the database")
            title = st.text_input("Title")
            artist = st.text_input("Artist")
            genre = st.text_input("Genre")
            tempo = st.number_input("Tempo", min_value=0.0)
            danceability = st.number_input("Danceability", min_value=0.0, max_value=1.0)
            energy = st.number_input("Energy", min_value=0.0, max_value=1.0)
            mood = st.text_input("Mood")
            acousticness = st.number_input("Acousticness", min_value=0.0, max_value=1.0)

            submitted = st.form_submit_button("Add Song")
            if submitted:
                if title and artist and genre:
                    new_id = recommender.add_song(
                        title=title,
                        artist=artist,
                        genre=genre,
                        tempo=tempo,
                        danceability=danceability,
                        energy=energy,
                        mood=mood,
                        acousticness=acousticness,
                    )
                    st.success(f"Song added successfully with ID: {new_id}")
                else:
                    st.error(
                        "Please fill in all required fields (Title, Artist, Genre)"
                    )


if __name__ == "__main__":
    main()
