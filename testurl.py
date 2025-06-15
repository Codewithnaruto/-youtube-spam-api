import requests

url = "http://127.0.0.1:5000/predict"
data = {"comment": "Check out my channel!", "author": "random_guy_42"}

res = requests.post(url, json=data)
print(res.json())
