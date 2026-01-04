import json
import sys

import boto3
import requests


def get_api_url(endpoint_name):
    client = boto3.client("apigateway")
    api_name = f"{endpoint_name}-api"

    apis = client.get_rest_apis()
    for item in apis.get("items", []):
        if item["name"] == api_name:
            api_id = item["id"]
            region = boto3.Session().region_name
            return f"https://{api_id}.execute-api.{region}.amazonaws.com/prod/predict"
    return None


def test_prediction():
    endpoint_name = "geo-busyness-endpoint"
    url = get_api_url(endpoint_name)

    if not url:
        print(f"Could not find API Gateway for endpoint: {endpoint_name}")
        print("Please ensure you have run deploy_model.py with CREATE_APIGW=true")
        return

    print(f"Testing API at: {url}")

    # Dummy data matching the 9 features expected by the model:
    # 1. dist_to_restaurant
    # 2. Hdist_to_restaurant
    # 3. avg_Hdist_to_restaurants
    # 4. date_day_number
    # 5. restaurant_id (encoded)
    # 6. Five_Clusters_embedding
    # 7. h3_index (encoded)
    # 8. date_hour_number
    # 9. restaurants_per_index

    # Example values (randomized but valid types)
    payload = "1.5,1.2,2.5,15,101,2,500,14,5"

    try:
        response = requests.post(
            url, data=payload, headers={"Content-Type": "text/csv"}
        )

        print(f"\nStatus Code: {response.status_code}")
        if response.status_code != 200:
            print(f"Response Body: {response.text}")
        else:
            print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_prediction()
