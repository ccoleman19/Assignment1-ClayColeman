#
# Author: Jamey Johnston
# Title: Test Flask App to Docker
# Date: 2020/01/30
# Email: jameyj@tamu.edu
# Texas A&M University - MS in Analytics - Mays Business School
#

import requests, json

# url = 'http://127.0.0.1:5000/winequality' # Local Flask Test
url = 'http://127.0.0.1:8081/banknote'  # Docker Test

 # "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"

text = json.dumps({"0": {"variance": 2.6, "skewness": 0.75, "kurtosis": 7.12, 'entropy': 0.23},
                   "1": {"variance": 3.0, "skewness": -0.75, "kurtosis": 3.12, 'entropy': -0.99},
                   "2": {"variance": -2.6, "skewness": 1.5, "kurtosis": -6.5, 'entropy': 0.0}})

headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

r = requests.post(url, data=text, headers=headers)

print(r, r.text)

