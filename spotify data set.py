# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 15:43:27 2025

@author: rowla
"""

import pandas as pd
import matplotlib.pyplot as plt

#importing csv from keggle
file_path = r"C:\Users\rowla\OneDrive\Desktop\Personal\uni work\BBE jaar 3\Big-data-design\3.UX and model building\Gastcollege langchain\Spotify\spotify_dataset.csv"
spotify_1 = pd.read_csv(file_path)

#cleaning the data set 
spotify = spotify_1.dropna()

#removing unneccesary columns 
columns_to_keep = [
    "Artist(s)",
    "song",
    "emotion",
    "Genre",
    "Album",
    "Tempo",
    "Energy",
    "Acousticness",
    "Danceability"
]

Spotify_2 = spotify[columns_to_keep]

# Count the number of each unique emotion
emotion_counts = Spotify_2["emotion"].value_counts()

# Group by Genre and Emotion, then count the occurrences
emotion_genre_counts = Spotify_2.groupby(["Genre", "emotion"]).size().unstack(fill_value=0)

# Create a condition for rows where:
# - Genre contains the word "rock" (case-insensitive)
# - Emotion is "anger"
condition = Spotify_2["Genre"].str.contains("rock", case=False, na=False) & (Spotify_2["emotion"] == "anger")

# Apply the new genre label "hard rock" to those rows
Spotify_2.loc[condition, "Genre"] = "hard rock"

# Create a condition for rows where:
# - Genre contains the word "rock" (case-insensitive)
# - Emotion is "anger"
condition = Spotify_2["Genre"].str.contains("rock", case=False, na=False) & (Spotify_2["emotion"] == "Love, confusion, fear, joy, love, sadness, surprise, thirst")

# Apply the new genre label "hard rock" to those rows
Spotify_2.loc[condition, "Genre"] = "rock"

# Find all rows where 'Genre' contains 'indie' (like 'indie rock', 'alt indie', etc.)
# and replace the value with just 'indie'
Spotify_2.loc[Spotify_2["Genre"].str.contains("indie", case=False, na=False), "Genre"] = "indie"

# Keep only specific genres
genres_to_keep = ["indie", "electronic", "hard rock", "rock"]
Spotify_3 = Spotify_2[Spotify_2["Genre"].isin(genres_to_keep)]

#changing emotion to mood to fit the recommender 
Spotify_clean = Spotify_3.rename(columns={"emotion": "mood"})

Spotify_clean.to_csv("spotify_cleaned.csv", index=False) 

# Count the number of songs per genre
genre_counts = Spotify_3["Genre"].value_counts()

# Create a pie chart with nice colors and labels with percentages
plt.figure(figsize=(8, 8))
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#ff6666', '#c2f0c2']  # Custom colors
genre_counts.plot(kind="pie", 
                  autopct='%1.1f%%',  # Display percentage in the chart
                  startangle=90,      # Start the chart from the top
                  colors=colors,      # Apply custom colors
                  wedgeprops={'edgecolor': 'black'},  # Adds edge color to wedges
                  figsize=(10, 10))

# Show the bar chart
plt.show()

# Create a colorful bar chart with reduced space between bars
plt.figure(figsize=(10, 6))

# Define a color palette
colors = plt.cm.Paired(range(len(genre_counts)))  # Automatically picks a range of colors

# Plot the bar chart with a reduced bar width
bars = genre_counts.plot(kind="bar", color=colors, edgecolor="black", width=0.8)  # width=0.8 reduces the space

# Add title and labels
plt.title("Distribution of Genres", fontsize=16)
plt.xlabel("Genre", fontsize=12)
plt.ylabel("Number of Songs", fontsize=12)

# Display the percentage labels on top of the bars
total = genre_counts.sum()
for i in range(len(genre_counts)):
    percentage = (genre_counts[i] / total) * 100
    bars.text(i, genre_counts[i] + 0.5, f'{percentage:.1f}%', ha='center', fontsize=10)

# Make sure the labels fit within the graph
plt.xticks(rotation=45)

# Adjust layout to avoid overlap
plt.tight_layout()

# Show the bar chart
plt.show()