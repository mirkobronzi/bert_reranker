import os
import json
import requests


# Export your gcloud auth token to an env variable
gcloud_auth_token = os.environ["GCLOUD_AUTH_TOKEN"]
data = {
    "instances": [
        {
            "input": "what is the main covid symptoms",
            "candidates": ["symptoms for covid", "hi", "can my employer fire me"],
        }
    ]
}
data = json.dumps(data)

url = "https://ml.googleapis.com/v1/projects/descartes-covid/models/hotline:predict?alt=json"
headers = {"Authorization": "Bearer " + gcloud_auth_token}

response = requests.post(url, data=data, headers=headers)
if response.status_code == 200:
    predictions = response.json()
