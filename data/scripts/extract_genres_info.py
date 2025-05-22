#!/usr/bin/env python3

import csv

with open("genres.csv") as f:
    genre_reader = csv.reader(f)
    header = genre_reader.__next__()

    for row in genre_reader:
        id = int(row[0])
        parent = int(row[2])
        title = row[3]

        if parent == 0:
            print(id, title)