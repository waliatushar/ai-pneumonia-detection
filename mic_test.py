import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print("Speak now...")
    audio = r.listen(source)

print("Recognizing...")
text = r.recognize_google(audio)
print("You said:", text)