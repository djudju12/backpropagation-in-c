#!/usr/bin/env python3

import csv
import json

with open("raw_tracks.csv") as f:
    track_reader = csv.reader(f)
    header = track_reader.__next__()
    # for i, h in enumerate(header):
    #     print(i, h)

    c = 0
    for row in track_reader:
        artist_name = row[5]
        track_date_recorded = row[20]
        track_file = row[26]

        if len(row[27]) > 0:
            track_genres = json.loads(row[27].replace("'", '"'))
        else:
            track_genres = []
        track_title = row[37]
        track_url = row[38]
        if len(track_genres) > 1:
            print(track_genres)
            break