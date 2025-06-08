# Note: By a "Song Embedding Model", we mean a set of embeddings which allow us to say "this song is simular to that
# song" - so it's like the word-similarity we just looked at, but we'll train it ourselves from a list of playlists
# each containing a list of songs (so each song is like a word in a sentence, and the playlist itself is the sentence!)

import os
import pandas as pd
from urllib import request
from gensim.models import Word2Vec

# File paths to save locally
TRAIN_FILE     = "DataFiles/train.txt"
SONG_HASH_FILE = "DataFiles/song_hash.txt"

# URLs to download from if local files don't exist
TRAIN_URL     = 'https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt'
SONG_HASH_URL = 'https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt'

# Download and save train.txt if it doesn't already exist
if not os.path.exists(TRAIN_FILE):
    print(f"Downloading {TRAIN_FILE}...")
    with request.urlopen(TRAIN_URL) as response, open(TRAIN_FILE, 'wb') as out_file:
        out_file.write(response.read())
else:
    print(f"Using cached {TRAIN_FILE}")

# Download and save song_hash.txt if it doesn't already exist
if not os.path.exists(SONG_HASH_FILE):
    print(f"Downloading {SONG_HASH_FILE}...")
    with request.urlopen(SONG_HASH_URL) as response, open(SONG_HASH_FILE, 'wb') as out_file:
        out_file.write(response.read())
else:
    print(f"Using cached {SONG_HASH_FILE}")

# Load the playlist data
with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')[2:]  # skip metadata lines

# Remove playlists with only one song
playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]

# Load the song metadata
with open(SONG_HASH_FILE, 'r', encoding='utf-8') as f:
    songs_file = f.read().split('\n')

songs = [s.rstrip().split('\t') for s in songs_file]

# Preview songs
print("----- First few songs -----")
for song in songs[:5]:
    print(song)

# Create a dataframe of all the songs using id/title/artist and make id the index
songs_dataframe = pd.DataFrame(data=songs, columns=['id', 'title', 'artist'])
songs_dataframe.set_index('id')

print("----- First few playlists -----")
for playlist in playlists[:5]:
    print(playlist)

# Train our Word2Vec model.
# Note: This takes a minute or so! Use `workers=1` AND `seed=<SOME_VALUE>` to get reproducible results!
model = Word2Vec(playlists, vector_size=32, window=20, negative=50, min_count=1, workers=4)  # seed=123)

def get_song_details(song_id: int):
    # Get the details of the row with the given song ID then build & return the string
    row = songs_dataframe.iloc[song_id]
    return f"id: {song_id} - artist: {row['artist']} - title: {row['title']}"


# Pick a song with a given id - 2172 is Metallica - Fade to Black
chosen_song_id      = 2172
chosen_song_details = get_song_details(chosen_song_id)
num_similar_songs   = 5

# Ask the model for songs similar to the one we've chosen
most_similar_songs: list[tuple[str, float]] = model.wv.most_similar(positive=str(chosen_song_id), topn=num_similar_songs)

# Print out the details
# Note: Because the most similar model is initially filled with noise, we get back slightly different results each time
# because they add or subtract from the noise, but the INITIAL NOISE VALUE might be high or low to be added or
# subtracted from. If we want the same results each time we can set `workers=1` AND `seed=123` or similar in the
# Word2Vec constructor, above and this will get us reproducible results (but as it's single threaded will take longer)!
print("----- Top " + str(num_similar_songs) + " similar songs to: " + str(chosen_song_id) + "(" + chosen_song_details + ") -----")
print(f"{'ID':<8} {'Artist':<20} {'Title':<30} {'Similarity':>10}")
for similar_song_id, similar_song_score in most_similar_songs:
    details = get_song_details(int(similar_song_id))
    # details is like: "id: 2849 - artist: Iron Maiden - title: Run To The Hills"
    # Let's split it out nicely by parsing or manual string split:
    # But assuming get_song_details returns string in that format, let's parse it:
    parts = details.split(' - ')
    # So parts will be like: ['id: 2849', 'artist: Iron Maiden', 'title: Run To The Hills']
    id_str     = parts[0].split(': ')[1]
    artist_str = parts[1].split(': ')[1]
    title_str  = parts[2].split(': ')[1]
    print(f"{id_str:<8} {artist_str:<20} {title_str:<30} {similar_song_score:10.6f}")