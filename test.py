import requests
import json

def test_api(endpoint):
    url = f"http://127.0.0.1:8000/{endpoint}"
    headers = {'Content-Type': 'application/json'}
    data = {"input_data": [5]} if endpoint == "predict" else {"training_data": [5, 10, 15]}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    print(response.status_code)
    if response.status_code == 500:
        print("Internal Server Error:", response.text)
    else:
        try:
            print(response.json())
        except requests.exceptions.JSONDecodeError:
            print("Response is not in JSON format")

if __name__ == "__main__":
    endpoint = input("Enter the endpoint to test (predict/train): ")
    test_api(endpoint)