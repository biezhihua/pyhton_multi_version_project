import requests

url = 'http://127.0.0.1:5000/compare'
files = {
    'image1': open('1.jpg', 'rb'),
    'image2': open('2.jpg', 'rb')
}

response = requests.post(url, files=files)
print('Status:', response.status_code)
print('Response:', response.json())
