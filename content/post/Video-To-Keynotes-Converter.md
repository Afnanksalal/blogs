---
title: "Turn Your Videos into Concise Keynotes: A Beginner's Guide to AI-Powered Video Summarization"
date: 2024-6-4
id: 2
author: "Afnan K Salal"
authorGithub: "https://github.com/afnanksalal"
tags:
  - Python
  - Streamlit
  - AI
  - Video Summarization
  - Google Cloud AI
  - Whisper
  - BART
---

##  Unlocking the Essence of Your Videos: Building an AI-Powered Video-to-Keynotes Generator

In today's information-packed world, video content is everywhere. Whether it's a webinar, a lecture, a podcast, or even a casual conversation, extracting the key points from these videos can be a daunting task. This is where AI comes to the rescue. In this comprehensive blog post, we'll build an intelligent application that can automatically summarize your videos and generate concise keynotes.  Get ready to learn how to harness the power of AI to make your video content more digestible and actionable.

###  Why Create a Video-to-Keynotes Generator?

* **Save Time and Effort:** Manually summarizing long videos can be time-consuming and tedious. Our AI-powered tool will do the heavy lifting for you.
* **Improved Comprehension:**  Keynotes provide a clear and concise overview of the video's main points, making it easier for viewers to grasp the essential information.
* **Enhanced Video Engagement:** By extracting keynotes, you can create summaries, highlights, or even transcripts to improve audience engagement.
* **Unlocking Value from Archived Content:**  Turn your existing video archives into valuable resources by summarizing them for easy access and reference.

###  Project Setup:  Setting the Stage

**1.  Create a Virtual Environment**

First, let's create a virtual environment to keep our project's dependencies separate from other Python projects. Open your terminal and follow these steps:

```bash
python3 -m venv my_video_env
source my_video_env/bin/activate
```

**2.  Install Required Libraries**

Now, install the necessary Python libraries using `pip`:

```bash
pip install streamlit transformers moviepy gTTS langchain langchain_google_genai pillow requests toml
```

*   **Streamlit:**  A Python library for building interactive web applications. It makes creating data-driven web apps super easy, even for beginners.
*   **Transformers:**  A library for working with state-of-the-art deep learning models, including those used for text summarization and speech recognition. 
*   **MoviePy:**  For processing video files and extracting audio.
*   **gTTS:**  A library for text-to-speech conversion.
*   **LangChain:**  A framework for building and interacting with AI applications.
*   **LangChain Google GenAI:**  A component of LangChain that lets you use Google Cloud AI models.
*   **Pillow:**  For image processing.
*   **Requests:**  Making API calls to web services.
*   **Toml:**  For handling configuration files in TOML format.

**3. Obtain API Keys**

We'll need an API key for Google Cloud AI:

*   **Google Cloud AI:**
    *   Sign up for a free trial of Google Cloud Platform ([https://cloud.google.com/](https://cloud.google.com/)).
    *   Enable the Google Cloud AI Platform API.
    *   Create a service account and obtain your API key.

**4. Set Up Environment Variables**

To store your API key securely, we'll use environment variables.

1.  **Create a `secrets.toml` File:**  In your project directory, create a file named `secrets.toml`.  This file will store your API key in a structured way. 
2.  **Add API Keys:** Add your API key to the `secrets.toml` file:

```toml
[google]
api_key = "YOUR_GOOGLE_API_KEY"
```

*   Replace `YOUR_GOOGLE_API_KEY` with the actual key you obtained.

**5.  Streamlit UI:  Your Application's Front End**

We'll use Streamlit to create an intuitive user interface for our video-to-keynotes generator. Here's the Streamlit code:

```python
import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import pipeline as transformers_pipeline
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from io import BytesIO
from moviepy.editor import VideoFileClip
import os
import toml
import tempfile

# --- Streamlit UI ---
st.title("Video to Keynotes Generator")

# --- Load Configuration ---
config = toml.load("secrets.toml")  # Load the TOML file

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    # --- Video to Audio ---
    video_bytes = uploaded_file.read()

    # Create a temporary file for the video
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_bytes)
        temp_file_path = temp_file.name

    video_clip = VideoFileClip(temp_file_path)
    audio_stream = video_clip.audio
    audio_path = "temp_audio.wav"  # Temporary audio file
    audio_stream.write_audiofile(audio_path)

    # --- Audio Transcription ---
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=438,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    result = pipe(audio_path)
    transcript = result["text"]

    st.subheader("Transcript")
    st.markdown(f"<pre>{transcript}</pre>", unsafe_allow_html=True)  # Use markdown for better formatting

    # --- Summarization ---
    summarizer = transformers_pipeline("summarization", model="facebook/bart-large-cnn")
    chunk_size = 1000
    max_length = 100  # You can adjust max_length as needed
    min_length = 30

    def summarize_chunks(text):
        """Summarizes text by splitting it into chunks and summarizing each chunk."""
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        return summaries

    summaries = summarize_chunks(transcript)
    combined_summary = "\n\n".join(summaries)

    st.subheader("Summary")
    st.markdown(f"<pre>{combined_summary}</pre>", unsafe_allow_html=True)  # Use markdown for better formatting

    # --- Keynotes Generation (using Gemini) ---
    GOOGLE_API_KEY = config["google"]["api_key"]  # Access the API key from config.toml

    def generate_keynotes_from_chunk(transcript_chunk):
        """Generates keynotes from a single chunk of transcript text."""
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=GOOGLE_API_KEY,
                max_output_tokens=100)
        except ValueError as e:
            st.error(f"Error initializing Gemini model: {e}")
            return None

        message = HumanMessage(
            content=f"Generate keynotes from the following transcript chunk, also don't use any beautifers like bold, italics, underscore, etc. Keep it plain text:\n\n{transcript_chunk}, also only accept the family friendly parts from the transcript and generate keynotes according to it. skip all unwanted and illegal parts for filtering"
        )
        try:
            response = llm.invoke([message])
            keynotes = response.content
            return keynotes
        except Exception as e:
            st.error(f"Error generating keynotes: {e}")
            return None

    chunk_size = 250
    transcript_chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)]

    all_keynotes = []
    for chunk in transcript_chunks:
        keynotes = generate_keynotes_from_chunk(chunk)
        if keynotes:
            all_keynotes.append(keynotes)

    # --- Display Keynotes in a Formatted Way ---
    st.subheader("Keynotes")
    for i, keynotes in enumerate(all_keynotes):
        st.write(keynotes)
        st.write("---")  # Add a separator between keynotes

    # --- Cleanup ---
    os.remove(audio_path)
    os.remove(temp_file_path)  # Delete the temporary file
```

*   **`st.title("Video to Keynotes Generator")`:**  Sets the title of the Streamlit app.
*   **`st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])`:**  Creates a file uploader component that accepts video files with the specified extensions. 

**6.  Video Processing:  From Video to Audio (using MoviePy)**

The first step is to extract the audio from the uploaded video using MoviePy.

```python
    # --- Video to Audio ---
    video_bytes = uploaded_file.read()

    # Create a temporary file for the video
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_bytes)
        temp_file_path = temp_file.name

    video_clip = VideoFileClip(temp_file_path)
    audio_stream = video_clip.audio
    audio_path = "temp_audio.wav"  # Temporary audio file
    audio_stream.write_audiofile(audio_path)
```

*   **`video_bytes = uploaded_file.read()`:** Reads the video file contents into memory.
*   **`tempfile.NamedTemporaryFile(delete=False)`:**  Creates a temporary file for storing the video data. The `delete=False` option prevents the file from being deleted automatically.
*   **`video_clip = VideoFileClip(temp_file_path)`:**  Loads the video using MoviePy.
*   **`audio_stream = video_clip.audio`:** Extracts the audio stream from the video clip.
*   **`audio_stream.write_audiofile(audio_path)`:**  Saves the audio stream to a temporary WAV file (`temp_audio.wav`).

**7.  Speech Recognition:  Turning Audio into Text**

Now, we'll use the Whisper speech recognition model from Hugging Face to convert the audio into text.

```python
    # --- Audio Transcription ---
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=438,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    result = pipe(audio_path)
    transcript = result["text"]

    st.subheader("Transcript")
    st.markdown(f"<pre>{transcript}</pre>", unsafe_allow_html=True)  # Use markdown for better formatting
```

*   **Device Selection:** Determines if the model should run on the GPU (if available) or the CPU.
*   **`AutoModelForSpeechSeq2Seq.from_pretrained`:**  Loads the Whisper model from Hugging Face.
*   **`AutoProcessor.from_pretrained`:**  Loads the processor associated with the Whisper model.
*   **`pipeline("automatic-speech-recognition", ...)`:**  Creates a pipeline for automatic speech recognition. 
*   **`result = pipe(audio_path)`:**  Runs the speech recognition pipeline on the audio file.
*   **`st.subheader("Transcript")`:**  Displays a subheader for the transcript in the Streamlit UI.
*   **`st.markdown(f"<pre>{transcript}</pre>", unsafe_allow_html=True)`:**  Displays the transcript in a formatted way using Markdown.

**8.  Summarization:  Extracting the Key Points**

We'll use the BART (Bidirectional Encoder Representations from Transformers) model to create summaries of the transcript.

```python
    # --- Summarization ---
    summarizer = transformers_pipeline("summarization", model="facebook/bart-large-cnn")
    chunk_size = 1000
    max_length = 100  # You can adjust max_length as needed
    min_length = 30

    def summarize_chunks(text):
        """Summarizes text by splitting it into chunks and summarizing each chunk."""
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        return summaries

    summaries = summarize_chunks(transcript)
    combined_summary = "\n\n".join(summaries)

    st.subheader("Summary")
    st.markdown(f"<pre>{combined_summary}</pre>", unsafe_allow_html=True)  # Use markdown for better formatting
```

*   **`transformers_pipeline("summarization", model="facebook/bart-large-cnn")`:**  Creates a pipeline for text summarization using the BART model.
*   **`summarize_chunks(text)`:**  A function to split the transcript into chunks and summarize each chunk individually.
*   **`summaries = summarize_chunks(transcript)`:**  Generates summaries for each chunk.
*   **`combined_summary = "\n\n".join(summaries)`:**  Combines the individual summaries into a single, comprehensive summary.
*   **`st.subheader("Summary")` and `st.markdown(...)`:**  Displays the summary in the Streamlit UI using Markdown.

**9.  Keynotes Generation:  Using Gemini for Insightful Summaries**

To generate keynotes from the transcript, we'll use the Gemini Pro model from Google Cloud AI. This model is excellent for understanding and extracting key points from text.

```python
    # --- Keynotes Generation (using Gemini) ---
    GOOGLE_API_KEY = config["google"]["api_key"]  # Access the API key from config.toml

    def generate_keynotes_from_chunk(transcript_chunk):
        """Generates keynotes from a single chunk of transcript text."""
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=GOOGLE_API_KEY,
                max_output_tokens=100)
        except ValueError as e:
            st.error(f"Error initializing Gemini model: {e}")
            return None

        message = HumanMessage(
            content=f"Generate keynotes from the following transcript chunk, also don't use any beautifers like bold, italics, underscore, etc. Keep it plain text:\n\n{transcript_chunk}, also only accept the family friendly parts from the transcript and generate keynotes according to it. skip all unwanted and illegal parts for filtering"
        )
        try:
            response = llm.invoke([message])
            keynotes = response.content
            return keynotes
        except Exception as e:
            st.error(f"Error generating keynotes: {e}")
            return None

    chunk_size = 250
    transcript_chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)]

    all_keynotes = []
    for chunk in transcript_chunks:
        keynotes = generate_keynotes_from_chunk(chunk)
        if keynotes:
            all_keynotes.append(keynotes)

    # --- Display Keynotes in a Formatted Way ---
    st.subheader("Keynotes")
    for i, keynotes in enumerate(all_keynotes):
        st.write(keynotes)
        st.write("---")  # Add a separator between keynotes

    # --- Cleanup ---
    os.remove(audio_path)
    os.remove(temp_file_path)  # Delete the temporary file
```

*   **`GOOGLE_API_KEY = config["google"]["api_key"]`:**  Retrieves the Google Cloud AI API key from the `secrets.toml` file.
*   **`generate_keynotes_from_chunk(transcript_chunk)`:**  A function to generate keynotes from a single chunk of transcript text.
    *   **`ChatGoogleGenerativeAI(model="gemini-pro", ...)`:** Initializes the Gemini Pro model.
    *   **`HumanMessage`:**  Creates a LangChain message object with a prompt that asks Gemini Pro to generate keynotes from the transcript chunk, ensuring plain text output and filtering out inappropriate content.
    *   **`response = llm.invoke([message])`:**  Invokes the Gemini Pro model with the prompt.
*   **Chunk Processing:**  The transcript is divided into chunks, and keynotes are generated for each chunk.
*   **Displaying Keynotes:**  The generated keynotes are displayed in a formatted way in the Streamlit UI.

**10.  Cleanup**

```python
    # --- Cleanup ---
    os.remove(audio_path)
    os.remove(temp_file_path)  # Delete the temporary file
```

*   **`os.remove(audio_path)` and `os.remove(temp_file_path)`:**  Deletes the temporary audio file and the temporary video file.

**11.  Running the Streamlit Application**

1.  Save your code in a Python file (e.g., `main.py`).
2.  Open your terminal and navigate to the directory where your `main.py` file is located.
3.  Run the Streamlit app:

```bash
streamlit run main.py
```

This will open your Streamlit app in a new browser window. Now you can upload a video, and the app will generate a transcript, summary, and keynotes.

###  Conclusion

You've successfully created an AI-powered video-to-keynotes generator using Python and Streamlit.  This application can help you quickly and effectively extract the most important information from your video content.

**Next Steps:**

*   **Customize the App:**  You can customize the UI, add more features, or integrate the keynotes generation into other workflows.
*   **Explore Different Models:**  Experiment with different language models for summarization and keynotes generation to see which ones work best for your needs.
*   **Deployment:**  Deploy your Streamlit app to a web server (like Heroku or Streamlit Cloud) to make it accessible to others.

This project demonstrates the incredible power of AI for summarizing and analyzing video content.  Enjoy exploring the possibilities!
---
