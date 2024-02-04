import speech_recognition as sr
import time
import vlc
from openai import OpenAI
import requests
import threading
from gtts import gTTS
from pygame import mixer
from bs4 import BeautifulSoup
import mediapipe as mp
from translate import get_translate
from transformers import pipeline
import cv2
from math import sqrt, acos, degrees
import random
import config

# Initialize OpenAI API key
client = OpenAI(api_key=config.openai_key)

# Classification command
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Initialize mixer for playing audio responses and music
mixer.init()

# Get urls
response = requests.get(config.radio_urls)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON data
    data = response.json()
    # Filter dictionaries where "source" is "2d"
    filtered_data = [entry for entry in data if entry["source"] == "2d"]
    if len(filtered_data) == 0:
        print("Some happens, music list is empty")
        exit()
    # Extract "name" and "url" from filtered dictionaries
    music_stations = [{"name": entry["name"].lower().replace("neural", "").replace("-", "").strip(), "url": entry["url"]} for entry in filtered_data]
    random.shuffle(music_stations)
else:
    print("Failed to fetch data from the URL. Check your internet connection. Status code:", response.status_code)
    exit()

# Authors TenChat
authors = {"Команда ТенЧат": "TenChat_admin", "ТенЧат для бизнеса": "tenchat_business", "Советы от Зевса": "zeus", "Семен Теняев": "tenchat", "Олег Ратунин": "oleg", "Владислав Радченко": "wladradchenko"}

# Commands to run by voice
commands = ["play music", "what news", "what are you created for", "other"]

# Initialize VLC player for streaming music
player = vlc.MediaPlayer(music_stations[0].get("url"))
is_speaking = False  # Global flag to indicate speaking state
music_thread = None  # Global variable to keep track of the music playing thread
news_thread = None


def play_music(url=None):
    if player.is_playing():
        player.stop()
    if url is not None:
        player.set_media(vlc.Media(url))
        player.play()
    else:
        text = get_translate("There is no such music on Neuron Radio. Write to the author about your wishes for new genres.", targetLang=config.assistant, sourceLang="en")
        speak(text, config.assistant)


def stop_music():
    global music_thread
    if player.is_playing():
        player.stop()
    music_thread = None


def pause_music():
    if player.is_playing():
        text = get_translate("The music has stopped.", targetLang=config.assistant, sourceLang="en")
        speak(text, config.assistant)
        player.stop()


def speak(text, lang='ru'):
    global is_speaking
    is_speaking = True  # Set flag to True when speaking starts
    phrases = text.split('. ')
    try:
        for phrase in phrases:
            if not is_speaking:  # Stop if speaking was interrupted
                break
            if phrase:  # Check if the phrase is not empty
                tts = gTTS(text=phrase, lang=lang, slow=False)
                tts.save("response.mp3")
                mixer.music.load("response.mp3")
                mixer.music.play()
                while mixer.music.get_busy():  # Wait for the phrase to finish playing
                    time.sleep(0.1)
                    if not is_speaking:  # Check if speaking should be stopped
                        mixer.music.stop()  # Stop the mixer if flag is False
                        break
    finally:
        is_speaking = False  # Reset flag when speaking is done or stopped


def stop_news():
    global is_speaking, news_thread
    if news_thread is not None:
        is_speaking = False  # Set the speaking flag to False to stop any ongoing speech synthesis
        news_thread = None  # Reset the news thread variable


def stop_speak():
    global is_speaking
    is_speaking = False
    stop_news()
    stop_music()


def parse_news(name="Команда ТенЧат", url="https://tenchat.ru/TenChat_admin"):
    # Parse random news
    text = get_translate(f"Get news by {name}", targetLang=config.assistant, sourceLang="en")
    speak(text, config.assistant)
    # Send a GET request to fetch the HTML content
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find all <a> elements with the specified class
        elements = soup.find_all("a", class_="tc-btn-focus group block w-full text-left")
        # Extract href attributes from the first 5 elements and store them in a list
        hrefs = [element.get("href") for element in elements[:5] if "/media/" in element.get("href")]
        # Random href
        random.shuffle(hrefs)
        if len(hrefs) > 0:
            href = hrefs[0]
            # Send a GET request to fetch the HTML content
            response = requests.get(f"https://tenchat.ru{href}")
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the HTML content using BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                # Find all elements with the specified class
                elements = soup.find_all(class_="grid grid-cols-1 gap-5 overflow-hidden")
                # Extract text from each element and store it in a list
                extracted_text = [element.get_text() + ". " for element in elements]
                # Join the extracted text from all elements into a single string
                page_text = '\n'.join(extracted_text)
                # Print the extracted text
                print(page_text)
                text = get_translate(page_text, targetLang=config.assistant, sourceLang="ru")
                speak(text, config.assistant)
            else:
                print("Failed to fetch data from the URL. Status code:", response.status_code)
    else:
        print("Failed to fetch data from the URL. Status code:", response.status_code)


def predict_command(user_command, default_commands):
    result = classifier(user_command, default_commands)
    # Extract the top category and its score
    top_category = result["labels"][0]
    top_category_score = result["scores"][0]
    # Confidence threshold for determining a close match
    confidence_threshold = 0.5
    # Handle the classification result
    if top_category == "other" or top_category_score < confidence_threshold:
        return "other"
    else:
        return top_category


def callback(recognizer, audio):
    global music_thread
    try:
        voice = recognizer.recognize_google(audio, language="ru-RU").lower()
        print("Распознано: " + voice)
        # If is word music in voice
        user_command = get_translate(voice, targetLang="en", sourceLang=config.assistant)
        predicted_command = predict_command(user_command, commands)

        if predicted_command == "play music":
            stop_news()
            if music_thread is None:
                # Get specific station
                for station in music_stations:
                    if station.get("name") in user_command:
                        genre_name = station.get("name")
                        genre_url = station.get("url")
                        break
                else:
                    genre_name = music_stations[0].get("name")
                    genre_url = music_stations[0].get("url")
                text = get_translate(f"Turn on {genre_name}.", targetLang=config.assistant, sourceLang="en")
                speak(text, config.assistant)
                music_thread = threading.Thread(target=play_music, args=(genre_url,))
                music_thread.start()
                print("Music started.")
            else:
                if not player.is_playing():
                    text = get_translate("Continue.", targetLang=config.assistant, sourceLang="en")
                    speak(text, config.assistant)
                    player.play()
                print("Music is already playing.")
        elif predicted_command == "what are you created for":
            stop_news()
            pause_music()
            text = get_translate("To turn on the Neural Radio and read TenChat!", targetLang=config.assistant, sourceLang="en")
            speak(text, config.assistant)
        elif predicted_command == "what news":
            # Get specific news
            author_names = [k.lower() for k in authors.keys()]
            predicted_author = predict_command(voice, author_names)
            for key, val in authors.items():
                if key.lower() == predicted_author:
                    news_name = key
                    news_url = val
                    break
            else:
                # Fallback to the first item in the dictionary if no match is found
                first_item = next(iter(authors.items()))  # Gets the first (key, value) pair
                news_name, news_url = first_item
            pause_music()
            news_thread = threading.Thread(target=parse_news, args=(news_name, f"https://tenchat.ru/{news_url}", ))
            news_thread.start()
        else:
            stop_news()
            pause_music()
            # Prepare the conversation history for GPT model
            text = get_translate(voice, targetLang="en", sourceLang=config.assistant)
            conversation_history = f"You are an intelligent assistant.\nHuman: {text}\nAI:"

            # Call the OpenAI API to generate a response
            response = client.completions.create(
                model="gpt-3.5-turbo",
                prompt=conversation_history,
                temperature=0.7,
                max_tokens=150,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )

            # Extract the reply from the response
            reply = response.choices[0].text.strip()
            print(f"ChatGPT: {reply}")

            # Use text-to-speech to speak the reply
            text = get_translate(reply, targetLang=config.assistant,  sourceLang="en")
            speak(text, config.assistant)
    except sr.UnknownValueError:
        print('Voice not recognized')
    except sr.RequestError:
        print('Unknown error!')


def handle_speech():
    r = sr.Recognizer()
    with sr.Microphone(device_index=4) as source:
        print("Adjusting for ambient noise, please wait...")
        r.adjust_for_ambient_noise(source)
        print("Listening...")
        text = get_translate("Listening.", targetLang=config.assistant, sourceLang="en")
        speak(text, config.assistant)
        audio = r.listen(source)
        callback(r, audio)


def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    ba = [a.x - b.x, a.y - b.y]
    bc = [c.x - b.x, c.y - b.y]
    cosine_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (sqrt(ba[0]**2 + ba[1]**2) * sqrt(bc[0]**2 + bc[1]**2))
    return degrees(acos(cosine_angle))


def detect_gesture():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    def recognize_gesture(landmarks):
        thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_mcp = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_mcp = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        pinky_mcp = landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
        # Angle between thumb tip, index tip, and middle MCP (for circle recognition)
        angle = calculate_angle(thumb_tip, index_tip, middle_mcp)
        # Calculate thumb-index tip distance
        thumb_index_distance = sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
        # Adjust threshold based on the scale observed in your application
        thumb_index_touching = thumb_index_distance < 0.05  # Adjusted threshold
        # Check for 'Index Up' gesture
        index_up = index_tip.y < index_mcp.y and all(tip.y > mcp.y for tip, mcp in [(middle_tip, middle_mcp), (ring_tip, ring_mcp), (pinky_tip, pinky_mcp)])
        if index_up:
            # print("Index Up")
            return False
        elif thumb_index_touching:
            # Check if other fingers are not clenched (i.e., extended away from palm)
            fingers_not_clenched = all(tip.y < mcp.y for tip, mcp in [(middle_tip, middle_mcp), (ring_tip, ring_mcp), (pinky_tip, pinky_mcp)])
            if fingers_not_clenched:
                if 85 <= angle <= 95:  # To reduce false okay
                    # print("Okay")
                    return True
                else:
                    return False
            else:
                # print("Unknown")
                return False
        else:
            if all(tip.y < mcp.y for tip, mcp in [(index_tip, index_mcp), (middle_tip, middle_mcp), (ring_tip, ring_mcp), (pinky_tip, pinky_mcp)]):
                # print("Open Hand")
                return False
            else:
                # print("Unknown")
                return False

    cap = cv2.VideoCapture(0)

    speech_thread = None  # Initialize the speech thread variable

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = recognize_gesture(hand_landmarks)
                if gesture and (speech_thread is None or not speech_thread.is_alive()):
                    stop_speak()  # Ensure this is thread-safe or suitable for your needs
                    speech_thread = threading.Thread(target=handle_speech)
                    speech_thread.start()

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


text = get_translate("Hello! I am the voice of TenChat. Show gesture Okay in order to I can start listening.", targetLang=config.assistant, sourceLang="en")
speak(text, config.assistant)

detect_gesture()
