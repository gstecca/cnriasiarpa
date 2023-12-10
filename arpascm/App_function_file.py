"""
proprietà Istituto di Analisi dei Sistemi ed Informatica "Antonio
Ruberti" del Consiglio Nazionale delle Ricerche

10/12/2023   versione v. 1.0

Autori:
Diego Maria Pinto
Marco Boresta
Giuseppe Stecca
Giovanni Felici
"""

import os
import warnings
import pandas as pd
warnings.simplefilter("ignore")
import random

def load_preprocessed_trajs_data(data_path, file_name):
    df = pd.read_excel(data_path + file_name, index_col=0)
    return  df

def random_color():
    return "#" + hex(random.randint(0, 0xFF))[2:] + hex(random.randint(0, 0xFF))[2:] + hex(random.randint(0, 0xFF))[2:]

def geo_df_to_timestamped_geojson(geo_df):
    color = random_color()
    df = geo_df.copy()
    x_coords = df["geometry"].x.to_list()
    y_coords = df["geometry"].y.to_list()
    coordinate_list = [[x, y] for x, y in zip(x_coords, y_coords)]
    times_list = [i.isoformat(sep='T', timespec='auto') for i in df.index.to_list()]

    feature_list = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinate_list,
            },
            "properties": {
                "times": times_list,
                "style": {
                    "color": color,
                    "weight": 5,
                },
            },
        }
    ]

    geojson = {}
    geojson['data'] = {
        'type': 'FeatureCollection',
        'features': feature_list
    }

    return geojson

