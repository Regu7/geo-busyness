import collections.abc
import logging
import os
from math import asin, cos, radians, sin, sqrt

import h3
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# File handler for logs
os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler("logs/pipeline.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def calc_dist(p1x, p1y, p2x, p2y):
    if isinstance(p1x, (list, tuple)):
        p1x = np.array(p1x)
        p1y = np.array(p1y)
        p2x = np.array(p2x)
        p2y = np.array(p2y)

    p1 = (p2x - p1x) ** 2
    p2 = (p2y - p1y) ** 2
    dist = np.sqrt(p1 + p2)
    return dist.tolist() if isinstance(dist, np.ndarray) else dist


def calc_haversine_dist(lat1, lon1, lat2, lon2):
    R = 6372.8
    if isinstance(lat1, collections.abc.Sequence):
        dLat = np.array([radians(l2 - l1) for l2, l1 in zip(lat2, lat1)])
        dLon = np.array([radians(l2 - l1) for l2, l1 in zip(lon2, lon1)])
        lat1 = np.array([radians(l) for l in lat1])
        lat2 = np.array([radians(l) for l in lat2])
    else:
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)

    a = np.sin(dLat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    dist = R * c
    return dist.tolist() if isinstance(lon1, collections.abc.Sequence) else dist


def initiate_centroids(k, df):
    centroids = df.sample(k)
    return centroids


def centroid_assignation(df, centroids):
    k = len(centroids)
    n = len(df)
    assignation = []
    assign_errors = []
    centroids_list = [c for i, c in centroids.iterrows()]
    for i, obs in df.iterrows():
        all_errors = [
            calc_dist(
                centroid["lat"], centroid["lon"], obs["courier_lat"], obs["courier_lon"]
            )
            for centroid in centroids_list
        ]
        nearest_centroid = np.where(all_errors == np.min(all_errors))[0].tolist()[0]
        nearest_centroid_error = np.min(all_errors)
        assignation.append(nearest_centroid)
        assign_errors.append(nearest_centroid_error)
    df["Five_Clusters_embedding"] = assignation
    df["Five_Clusters_embedding_error"] = assign_errors
    return df


def Encoder(df):
    columnsToEncode = list(df.select_dtypes(include=["category", "object"]))
    le = LabelEncoder()
    encoders = {}
    for feature in columnsToEncode:
        try:
            df[feature] = le.fit_transform(df[feature])
            encoders[feature] = le
        except:
            print("Error encoding " + feature)
    return df, encoders


def generate_features(df, restaurants_ids):
    config = load_config()
    resolution = config["h3_resolution"]
    k = config["k_clusters"]
    seed = config["random_seed"]

    logger.info("Starting feature generation")

    # Euclidean distance to restaurant
    df["dist_to_restaurant"] = calc_dist(
        df.courier_lat, df.courier_lon, df.restaurant_lat, df.restaurant_lon
    )

    # Avg euclidean distance to restaurants
    def avg_dist_to_restaurants(courier_lat, courier_lon):
        return np.mean(
            [
                calc_dist(v["lat"], v["lon"], courier_lat, courier_lon)
                for v in restaurants_ids.values()
            ]
        )

    df["avg_dist_to_restaurants"] = [
        avg_dist_to_restaurants(lat, lon)
        for lat, lon in zip(df.courier_lat, df.courier_lon)
    ]

    # Haversine distance to restaurant
    df["Hdist_to_restaurant"] = calc_haversine_dist(
        df.courier_lat.tolist(),
        df.courier_lon.tolist(),
        df.restaurant_lat.tolist(),
        df.restaurant_lon.tolist(),
    )

    # Avg Haversine distance to restaurants
    def avg_Hdist_to_restaurants(courier_lat, courier_lon):
        return np.mean(
            [
                calc_haversine_dist(v["lat"], v["lon"], courier_lat, courier_lon)
                for v in restaurants_ids.values()
            ]
        )

    df["avg_Hdist_to_restaurants"] = [
        avg_Hdist_to_restaurants(lat, lon)
        for lat, lon in zip(df.courier_lat, df.courier_lon)
    ]

    # Five-Clusters embedding
    np.random.seed(seed)
    df_restaurants = pd.DataFrame(
        [{"lat": v["lat"], "lon": v["lon"]} for v in restaurants_ids.values()]
    )
    centroids = initiate_centroids(k, df_restaurants)
    df = centroid_assignation(df, centroids)

    # H3 clustering
    df["courier_location_timestamp"] = pd.to_datetime(
        df["courier_location_timestamp"], format="ISO8601"
    )
    df["order_created_timestamp"] = pd.to_datetime(
        df["order_created_timestamp"], format="ISO8601"
    )
    df["h3_index"] = [
        h3.latlng_to_cell(lat, lon, resolution)
        for (lat, lon) in zip(df.courier_lat, df.courier_lon)
    ]
    df["date_day_number"] = [d for d in df.courier_location_timestamp.dt.day_of_year]
    df["date_hour_number"] = [d for d in df.courier_location_timestamp.dt.hour]

    # Orders busyness
    index_list = [
        (i, d, hr)
        for (i, d, hr) in zip(df.h3_index, df.date_day_number, df.date_hour_number)
    ]
    set_indexes = list(set(index_list))
    dict_indexes = {label: index_list.count(label) for label in set_indexes}
    df["orders_busyness_by_h3_hour"] = [dict_indexes[i] for i in index_list]

    # Restaurants per index
    restaurants_counts_per_h3_index = {
        a: len(b)
        for a, b in zip(
            df.groupby("h3_index")["restaurant_id"].unique().index,
            df.groupby("h3_index")["restaurant_id"].unique(),
        )
    }
    df["restaurants_per_index"] = [
        restaurants_counts_per_h3_index[h] for h in df.h3_index
    ]

    # Label encoding
    df["h3_index"] = df.h3_index.astype("category")
    df, encoders = Encoder(df)

    logger.info("Feature generation completed")

    artifacts = {
        "centroids": centroids,
        "restaurants_counts_per_h3_index": restaurants_counts_per_h3_index,
        "dict_indexes": dict_indexes,
        "encoders": encoders,
    }
    return df, artifacts


if __name__ == "__main__":
    from data_ingestion import load_and_process_data

    df, restaurants = load_and_process_data()
    df = generate_features(df, restaurants)
    logger.info(f"Processed dataset shape: {df.shape}")
    logger.info(f"Sample features: {list(df.columns)}")
    logger.info(f"Number of unique restaurants: {len(restaurants)}")
