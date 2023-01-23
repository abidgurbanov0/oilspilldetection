import requests
import json

multipart_form_data = {
    'file': ('3.jpg', open('test.jpg', 'rb')),
}
response = requests.post('http://127.0.0.1:8000/api/v1/upload', files=multipart_form_data)
print(response.status_code,json.loads(response.content))