[![Price](https://img.shields.io/badge/price-FREE-0098f7.svg)](https://github.com/wladradchenko/assistant.wladradchenko.ru/blob/main/LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![GitHub package version](https://img.shields.io/github/v/release/wladradchenko/assistant.wladradchenko.ru?display_name=tag&sort=semver)](https://github.com/wladradchenko/assistant.wladradchenko.ru)
[![License: MIT v1.0](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/wladradchenko/assistant.wladradchenko.ru/blob/main/LICENSE)

<p align="right">(<a href="README_ru.md">RU</a>)</p>
<div id="top"></div>

<br />
<div align="center">
  <a href="https://github.com/wladradchenko/assistant.wladradchenko.ru">
    <img src="https://media.giphy.com/media/fO3jej3pv2ofGkquhO/giphy.gif" alt="Logo" width="200" height="200">
  </a>
  
  <h3 align="center">Anniversary Voice Assistant</h3>

  <p align="center">
    Documentation
    <br/>
    <br/>
    <br/>
    <a href="https://github.com/wladradchenko/assistant.wladradchenko.ru/issues">Report a Bug</a>
    ·
    <a href="https://github.com/wladradchenko/assistant.wladradchenko.wladradchenko.ru/issues">Request a Feature</a>
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About the Project

The Voice Assistant is an anniversary project in honor of 10,000 subscribers in [TenChat](https://tenchat.ru/wladradchenko) blog. The project is suitable for integration with Jetson Nano, Raspberry Pi 4, and Pi 5. To activate the voice assistant, make the "Okay" gesture, and then say "Listening" followed by your request.

The project's goal is to create a voice assistant that includes "Neural Radio," reads posts from authors in TenChat, or asks questions to ChatGPT. Working with ChatGPT requires an OpenAI API key.

Main Features:

- Activation with the "Okay" gesture.
- Ask to play any genre of music from "Neural Radio" or explore new neural music. Anyway, you can ask the voice assistant what music genres are available on Neural Radio.
- Ask to read the latest news from your favorite author in TenChat. New authors can be added to the list.
- Make a request to ChatGPT.

__P.S. Normal app functionality requires an internet connection to work with Google and OpenAI.__

The application includes:

- Mediapipe for hand gesture recognition.
- Transformers for text processing.
- gTTS for simple text-to-speech.
- SpeechRecognition for voice recognition from the microphone.

Join and listen to "Infinite Neural Radio" on [GitHub](https://github.com/wladradchenko/assistant.wladradchenko.ru), [YouTube](https://www.youtube.com/@wladradchenko), and on the [official website](https://radio.wladradchenko.ru).

## Setup
Requirements Python version 3.10.

Create venv

```
python3.10 -m venv venv
```

Activate venv

```
// Linux / MacOS
source venv/bin/activate
// Windows
venv\\Scripts\\activate.bat
```

Install requirements

```
python -m pip install -r requirements.txt
```

Run

```
python main.py
```

<!-- DONATION -->
## Support

You can support the project's author in developing creative ideas or simply buy them a [cup of coffee](https://wladradchenko.ru/donat).
<!-- DONATION -->

<!-- CONTACT -->
## Contact

Authors: 
- [Wladislav Radchenko](https://github.com/wladradchenko/)

Email: [i@wladradchenko.ru](i@wladradchenko.ru)

Project: [Anniversary Voice Assistant](https://assistant.wladradchenko.ru)

<p align="right">(<a href="#top">back to top</a>)</p>
