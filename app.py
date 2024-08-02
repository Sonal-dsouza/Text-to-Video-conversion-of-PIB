import os
from flask import Flask, render_template, request, send_file
import requests
import nltk
from gtts import gTTS
from moviepy.editor import TextClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip, ImageClip
from pyunsplash import PyUnsplash
from summa import summarizer
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import fitz  # PyMuPDF
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter

app = Flask(__name__)

# Set the path to the ImageMagick binary
os.environ["IMAGE_MAGICK_BINARY"] = r"C:\Users\sonal\OneDrive\Desktop\ImageMagick Display.lnk"  # Adjust the path accordingly

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Initialize PyUnsplash
pu = PyUnsplash(api_key="004j7OBNyfHjNwaQ9DOUxDxl8UMXQyNPI_7ec8FVF8Q")

# Initialize SpaCy
nlp = spacy.load("en_core_web_sm")

# Function to extract keywords
def extract_keywords(text, num_keywords=10):
    doc = nlp(text)
    keywords = []

    for token in doc:
        if token.text.lower() not in STOP_WORDS and not token.is_punct:
            if token.pos_ in ["NOUN", "PROPN", "ADJ"]:  # Consider only nouns, proper nouns, and adjectives
                keywords.append(token.lemma_.lower())

    # Extract named entities
    for ent in doc.ents:
        if ent.label_ in ["LOC", "ORG"]:  # Consider locations and organizations
            keywords.append(ent.text.lower())

    # Extract keywords using TextRank
    phrase_list = [phrase.text for phrase in doc.noun_chunks]
    phrase_freq = Counter(phrase_list)
    max_freq = max(phrase_freq.values(), default=1)
    for phrase in phrase_list:
        keywords.append(phrase.lower())  # Add noun phrases

    # Remove duplicates
    keywords = list(set(keywords))

    # Limit to num_keywords
    keywords = keywords[:num_keywords]

    print("Extracted Keywords:", keywords)
    return keywords

# Function to fetch images from Unsplash
def fetch_unsplash_images(keywords, count=10, image_size=(800, 600)):
    query = " ".join(keywords)
    photos = pu.photos(type_='random', count=count, featured=True, query=query)
    image_urls = [photo.link_download for photo in photos.entries]
    return [(url, image_size) for url in image_urls]

# Function to download images
def download_images(image_data, output_folder="images"):
    os.makedirs(output_folder, exist_ok=True)
    downloaded_images = []

    for i, (url, size) in enumerate(image_data):
        response = requests.get(url, allow_redirects=True)
        image_path = os.path.join(output_folder, f"image_{i + 1}.jpg")
        with open(image_path, 'wb') as f:
            f.write(response.content)
        downloaded_images.append(image_path)

    return downloaded_images

# Function to resize images
def resize_images(image_paths, output_size=(800, 600)):
    resized_images = []

    for img_path in image_paths:
        with Image.open(img_path) as img:
            img_resized = img.resize(output_size)
            resized_path = os.path.splitext(img_path)[0] + "_resized.jpg"
            img_resized.save(resized_path)
            resized_images.append(resized_path)

    return resized_images

# Function to generate audio from text
def generate_audio(summary, output_filename="summary_audio.mp3"):
    tts = gTTS(summary)
    tts.save(output_filename)

# Function to generate video
def generate_video(images, summary, audio_filename="summary_audio.mp3", output_filename="output.mp4"):
    image_duration = 10
    summary_clip = TextClip(summary, fontsize=24, color="white", size=(800, None), method="caption", align="center")
    summary_clip = summary_clip.set_duration(len(images) * image_duration)

    image_clips = [ImageClip(image).set_duration(image_duration) for image in images]
    final_clip = concatenate_videoclips(image_clips, method="compose")
    final_clip = CompositeVideoClip([final_clip.set_duration(summary_clip.duration), summary_clip.set_pos("bottom")])

    audio_clip = AudioFileClip(audio_filename)
    final_clip = final_clip.set_audio(audio_clip)

    output_path = os.path.join(os.getcwd(), output_filename)
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=25)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf_file:
        for page in pdf_file:
            text += page.get_text()
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pdf_file = request.files['pdf_file']
        if pdf_file.filename != '':
            pdf_filename = pdf_file.filename
            pdf_path = os.path.join(app.root_path, 'uploads', pdf_filename)
            pdf_file.save(pdf_path)

            # Extract text from PDF
            pdf_text = extract_text_from_pdf(pdf_path)

            # Summarize the extracted text
            summary = summarizer.summarize(pdf_text, ratio=0.6)

            # Extract keywords from the summary
            keywords = extract_keywords(summary)

            # Fetch images related to keywords from Unsplash
            image_data = fetch_unsplash_images(keywords, count=len(keywords))

            # Download and resize images
            downloaded_images = download_images(image_data)
            resized_images = resize_images(downloaded_images)

            # Generate audio from the summary
            generate_audio(summary)

            # Generate video
            generate_video(resized_images, summary)

            return render_template('result.html', video_filename='output.mp4')

    return render_template('index.html')

@app.route('/download_video')
def download_video():
    video_path = os.path.join(app.root_path, 'output.mp4')
    return send_file(video_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
