# Nateq - Intelligent Arabic Text-to-Speech with OpenAI

Nateq - (ناطق) is an innovative application that leverages artificial intelligence solutions, specifically designed for Arabic language processing. Tailored for Arabicthon 2023, ناطق combines the power of OpenAI's language models and Google Cloud Vision API to offer advanced text-to-speech capabilities.

## Features

- **Fast Text-to-Speech Conversion:** Utilizes OpenAI's powerful language models to generate high-quality Arabic speech from text.
- **Enhanced Audio:** Applies audio processing techniques to improve the quality of generated speech.
- **Image-Enhanced Text Recognition:** Integrates with Google Cloud Vision API for accurate Arabic text extraction from images.
- **Interactive Interface:** Provides a user-friendly interface for users to input text, upload images, and record their own pronunciations.

## Installation

To run Nateq locally, follow these steps:

1. Install the required Python packages:

   ```bash
   pip install gradio pydub librosa google-cloud-vision numpy
   ```

2. Install the Tesseract OCR engine:

* On Linux:
```
sudo apt-get install tesseract-ocr
```

* On macOS:
```
brew install tesseract
```

3. Install the required Python packages for Tesseract OCR:
```
pip install pytesseract
```

4. Set up Google Cloud credentials:

* Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of your Google Cloud Vision API key file.

5. Run the (Nateq) application:
```
python Nateq.py
```

## Usage
(Nateq) offers multiple functionalities through its intuitive interface:

*Faster Model*: Use the faster model for quick text-to-speech conversion.
*Better Model*: Opt for the better model when quality is a priority.
*Record!*: Record your own pronunciation and receive a thank-you message.

## Contributing
If you'd like to contribute to (Nateq), feel free to open an issue or submit a pull request. Your feedback and enhancements are highly appreciated.