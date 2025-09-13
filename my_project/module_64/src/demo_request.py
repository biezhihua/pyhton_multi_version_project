
import requests

url = 'http://127.0.0.1:5000/compare'
with open('1.jpg', 'rb') as f1, open('2.jpg', 'rb') as f2:
    files = {
        'image1': ('1.jpg', f1, 'image/jpeg'),
        'image2': ('2.jpg', f2, 'image/jpeg')
    }
    response = requests.post(url, files=files)
    print('Status:', response.status_code)
    print('Response:', response.json())
