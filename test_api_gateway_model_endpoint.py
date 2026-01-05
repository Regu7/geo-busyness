import json

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
    endpoint_name = "geo-busyness-endpoint-prod"
    url = get_api_url(endpoint_name)

    if not url:
        print(f"Could not find API Gateway for endpoint: {endpoint_name}")
        print("Please ensure you have run deploy_model.py with CREATE_APIGW=true")
        return

    print(f"Testing API at: {url}")

    # Raw input data expected by the model inference pipeline
    # Columns: courier_lat, courier_lon, restaurant_lat, restaurant_lon, courier_location_timestamp

    # Example data (CSV format with headers)
    payload = """courier_lat,courier_lon,restaurant_lat,restaurant_lon,courier_location_timestamp
50.484520325268576,-104.6188756,50.483696253259296,-104.6143496,2021-04-02T04:30:42.328Z
50.44257272227587,-104.5504633,50.4424223,-104.5504874,2021-04-01T06:14:47.386Z"""

    try:
        # We must specify Accept: application/json to get a JSON response.
        # Otherwise, it defaults to Content-Type (text/csv) and returns a CSV string,
        # which causes response.json() to fail.
        response = requests.post(
            url,
            data=payload,
            headers={"Content-Type": "text/csv", "Accept": "application/json"},
        )

        print(f"\nStatus Code: {response.status_code}")
        print(f"Raw Response: {response.text}")

        if response.status_code == 200:
            try:
                print(f"Response JSON: {response.json()}")
            except json.JSONDecodeError:
                # print("Response is not valid JSON (likely CSV).")
                print(f"Response Text: {response.text}")
        else:
            print(f"Response Body: {response.text}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_prediction()
