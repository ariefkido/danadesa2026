from google import genai

client = genai.Client(api_key="AIzaSyCugcOSn_fUXa1o_yPm7ByM3BGJFmCDe6A")

for model in client.models.list():
    print(model.name)