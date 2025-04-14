import streamlit as st
from recommender import SongRecommender

# Initialize the recommender system
@st.cache_resource
def load_recommender():
    return SongRecommender()

recommender = load_recommender()

# Streamlit app
def main():
    st.title("🎵 Song Recommender System")
    st.write("Discover songs similar to your favorites!")
    # Search for a song
    st.subheader("Search for a song")
    search_query = st.text_input("Enter a song title or artist:")

    if search_query:
        # Find matching songs
        matches = recommender.search_songs(search_query)
        
        if not matches.empty:
            # Display matches and let user select one
            st.write("Matching songs found:")
            matches_display = matches[["song_id", "title", "artist", "genre", "tempo","danceability", "mood"]].set_index("song_id")
            # display matches in a scrollable table
            st.dataframe(matches_display, use_container_width=True)
            
            # Let user select a song by ID
            selected_id = st.selectbox(
                "Select a song for recommendations:",
                options=matches['song_id'].tolist(),
                format_func=lambda x: f"{recommender.get_song_by_id(x).iloc[0]['title']} by {recommender.get_song_by_id(x).iloc[0]['artist']}"
            )
            
            if st.button("Get Recommendations"):
                selected_song = recommender.get_song_by_id(selected_id).iloc[0]
                st.success(f"Selected: **{selected_song['title']}** by {selected_song['artist']}")
                st.table(selected_song[["song_id", "title", "artist", "genre", "tempo", "danceability", "mood"]].to_frame().T.set_index("song_id"))
                
                # Get recommendations
                with st.spinner("Finding similar songs..."):
                    recommendations = recommender.recommend([selected_id])
                
                if isinstance(recommendations, str):
                    st.warning(recommendations)
                else:
                    st.subheader("Recommended songs:")
                    st.table(recommendations.set_index("song_id")[["title", "artist", "genre", "tempo", "danceability", "mood"]])
        else:
            st.warning("No matching songs found. Please try another search.")


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
                        acousticness=acousticness
                    )
                    st.success(f"Song added successfully with ID: {new_id}")
                else:
                    st.error("Please fill in all required fields (Title, Artist, Genre)")

if __name__ == "__main__":
    main()