import pandas as pd
import random

data = []

for i in range(3000):

    fever = random.choice([0,1])
    cough = random.choice([0,1])
    breath_shortness = random.choice([0,1])
    chest_pain = random.choice([0,1])
    fatigue = random.choice([0,1])
    sore_throat = random.choice([0,1])
    body_ache = random.choice([0,1])
    smoking = random.choice([0,1])
    diabetes = random.choice([0,1])
    age = random.randint(5,90)

    # ---- pneumonia logic ----
    score = fever + cough + breath_shortness + chest_pain

    if score >= 3 or (age > 65 and cough == 1) or (diabetes == 1 and fever == 1):
        pneumonia = 1
    else:
        pneumonia = 0

    data.append([
        fever, cough, breath_shortness, chest_pain,
        fatigue, sore_throat, body_ache,
        smoking, diabetes, age, pneumonia
    ])

df = pd.DataFrame(data, columns=[
    "fever","cough","breath_shortness","chest_pain",
    "fatigue","sore_throat","body_ache",
    "smoking","diabetes","age","pneumonia"
])

df.to_csv("pneumonia_dataset_3000.csv", index=False)

print("✅ Dataset created successfully with 3000 rows")