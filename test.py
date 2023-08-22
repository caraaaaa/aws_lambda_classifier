import requests

# for testing in local
#url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# for testing the API Gateway
aws_account_id = ""
region = ""
repo_name = ""
HOST = f"https://{aws_account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}"
url = f'{HOST}/predict' 

# Choose your own image
img_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Smaug_par_David_Demaret.jpg/1280px-Smaug_par_David_Demaret.jpg'
data = {"url": img_url}

result = requests.post(url, json=data)
print(result.json())