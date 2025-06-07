import requests

url = "https://webdav.yandex.ru/data"
auth = ("alikDavletshin", "faysapbhhwagxgek")

response = requests.request("MKCOL", url, auth=auth)
print(response.status_code)  # 201 - успешно создана