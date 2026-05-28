import google.generativeai as genai

genai.configure(api_key="AIzaSyA-qNq3tsd_9qYtbv0bXmIjRf2aYJqNYxs")

models = genai.list_models()

for m in models:
    print(m.name, " -> ", m.supported_generation_methods)