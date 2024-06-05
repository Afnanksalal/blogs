---
title: "Text-to-Speech with Tone Transfer"
date: 2024-6-6
id: 4
author: "Afnan K Salal"
authorGithub: "https://github.com/afnanksalal"
tags:
  - Python
  - Voice Cloning
  - OpenVoice
  - MeloTTS
  - Text-to-Speech
  - AI
  - Streamlit
---

##  Unleash the Power of Voice Cloning: Building a Text-to-Speech App with OpenVoice and MeloTTS

Imagine this: you want to create an engaging audio narration for your video, but you don't have a professional voice actor. Or perhaps you'd like to generate audio messages in the voice of your favorite celebrity.  With voice cloning technology, these scenarios are becoming a reality.

This blog post will guide you through creating a text-to-speech app that leverages the capabilities of OpenVoice and MeloTTS to clone voices and generate speech with transferred tone. 

### What is Voice Cloning?

Voice cloning is an AI technology that allows you to recreate someone's voice using machine learning. By training a model on audio recordings of a target speaker, you can generate new audio that sounds remarkably similar to the original voice.

### Why Use Voice Cloning?

Voice cloning offers a range of exciting possibilities, from creative applications to practical uses:

*   **Content Creation:**  Add professional-sounding narration to videos, podcasts, and audiobooks. 
*   **Interactive Storytelling:**  Create immersive experiences by having characters speak in cloned voices.
*   **Accessibility:**  Make written content accessible to visually impaired individuals.
*   **Entertainment:**  Generate personalized messages or parodies in the voices of celebrities or friends.
*   **Personalized Assistants:**  Create AI assistants with voices that are more engaging and human-like.

### Let's Get Started: Setting Up Your Project

**1.  Project Setup:**

**1.1 Create a Virtual Environment**

As a good practice, we'll create a virtual environment to manage our project's dependencies.

1.  Open your terminal and create a new virtual environment:
    ```bash
    python3 -m venv my_voice_env
    ```
2.  Activate the virtual environment:
    ```bash
    source my_voice_env/bin/activate
    ```

**1.2 Install Required Libraries**

*   **OpenVoice:**  A powerful open-source library for voice cloning and tone transfer. 
*   **MeloTTS:**  A robust text-to-speech library.
*   **pydub:**  For audio file manipulation.
*   **streamlit:**  For creating a simple web interface for our app.

**2. OpenVoice Setup**

**2.1 Download OpenVoice & Checkpoint**

*   Download the OpenVoice from the official repository ([https://github.com/myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice)).
*   Download the OpenVoice Checkpoint ([https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip](https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip)).
*   Extract the downloaded checkpoint zip file to a folder named `checkpoint` in your project directory.

**2.2 Install MeloTTS**

*   Install MeloTTS from the GitHub repository:
    ```bash
    pip install git+https://github.com/myshell-ai/MeloTTS.git
    ```
*   Download language data:
    ```bash
    python -m unidic download
    ```

**3. Understanding the Code: Decoding `main.py`**

Now, let's dive into the code, line by line, to understand how our voice cloning app works.

```python
import io
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
from pydub import AudioSegment
import streamlit as st
import os
from tempfile import NamedTemporaryFile

# --- OpenVoice Setup ---
ckpt_converter = os.path.join(os.path.expanduser('~'), 'AudioGenerator', 'checkpoint', 'converter')
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # --- Streamlit UI ---
    st.set_page_config(layout="wide", page_title="Text-to-Speech with Tone Transfer")
    st.title("Text-to-Speech with Tone Transfer")

    # Text Input
    text_input = st.text_area("Enter your text:", height=200)

    # Language Selection
    language = st.selectbox("Select Language:", ["EN_NEWEST", "ES", "FR", "ZH", "JP", "KR"])

    # Reference Audio Upload
    uploaded_file = st.file_uploader("Upload Reference Audio (wav or mp3):", type=["wav", "mp3"])

    # --- Audio Generation and Tone Transfer ---
    if st.button("Generate Speech"):
        with st.spinner("Generating audio..."):
            if uploaded_file is not None:
                if not generate_speech(text_input, language, uploaded_file):
                    return
            else:
                st.warning("Please upload a reference audio file.")

def generate_speech(text_input, language, uploaded_file):
    try:
        # Load the tone color converter
        tone_color_converter = ToneColorConverter(os.path.join(ckpt_converter, 'config.json'), device=device)
        tone_color_converter.load_ckpt(os.path.join(ckpt_converter, 'checkpoint.pth'))
    except FileNotFoundError:
        st.error("Error: 'config.json' or 'checkpoint.pth' not found. Please make sure the files exist in the correct location.")
        return False

    try:
        # Read uploaded audio file
        audio_data = uploaded_file.read()
        audio_format = uploaded_file.type.split('/')[1]

        # Convert MP3 to WAV if necessary
        if audio_format == 'mp3':
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
            with NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
                audio_segment.export(temp_wav_file.name, format='wav')
                temp_wav_file_name = temp_wav_file.name
        else:
            with NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
                temp_wav_file.write(audio_data)
                temp_wav_file_name = temp_wav_file.name

        # Extract target speaker embedding from reference audio
        target_se, audio_name = se_extractor.get_se(temp_wav_file_name, tone_color_converter, vad=False)
    except Exception as e:
        st.error(f"Error extracting speaker embedding from reference audio: {e}")
        return False

    try:
        model = TTS(language=language, device=device)
        speaker_ids = model.hps.data.spk2id
        audio_segments = []

        for speaker_key in speaker_ids.keys():
            speaker_id = speaker_ids[speaker_key]
            speaker_key = speaker_key.lower().replace('_', '-')

            source_se = torch.load(os.path.join(os.path.expanduser('~'), 'AudioGenerator', 'checkpoint', 'base_speakers', 'ses', f'{speaker_key}.pth'), map_location=device)

            # Split text into sentences
            sentences = text_input.split('.')

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                try:
                    # Generate audio in memory (using io.BytesIO)
                    with NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
                        model.tts_to_file(sentence, speaker_id, temp_audio_file.name, speed=1.0)

                        audio_segment = AudioSegment.from_wav(temp_audio_file.name)

                        # Apply tone color conversion directly
                        with NamedTemporaryFile(delete=False, suffix='.wav') as output_temp_file:
                            tone_color_converter.convert(
                                audio_src_path=temp_audio_file.name,
                                src_se=source_se,
                                tgt_se=target_se,
                                output_path=output_temp_file.name,
                                message="@MyShell"
                            )

                            # Replace the original audio segment with the tone-converted one
                            audio_segment = AudioSegment.from_wav(output_temp_file.name)
                            audio_segments.append(audio_segment)

                except Exception as e:
                    st.error(f"Error during processing: {e}")
                    st.error("Trying to recover...")

            if audio_segments:
                # Merge audio segments for the current speaker
                merged_audio = audio_segments[0]
                for i in range(1, len(audio_segments)):
                    merged_audio += audio_segments[i]

                # Save the merged audio to a NamedTemporaryFile
                with NamedTemporaryFile(delete=False, suffix='.wav') as output_temp_file:
                    merged_audio.export(output_temp_file.name, format="wav")

                    # Download the audio in Streamlit
                    st.download_button(
                        label="Download Audio",
                        data=open(output_temp_file.name, 'rb'),
                        file_name=f"output_v2_{speaker_key}.wav",
                        mime="audio/wav"
                    )

                # Reset audio segments for the next speaker
                audio_segments = []

        st.success("Speech generation complete!")
        return True
    except Exception as e:
        st.error(f"Error during processing: {e}")
        st.error("Trying to recover...")
        return False

if __name__ == "__main__":
    main()
```

**3.1  Import Statements**

```python
import io
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
from pydub import AudioSegment
import streamlit as st
import os
from tempfile import NamedTemporaryFile
```

*   **`io`:**  Provides tools for working with input/output streams (e.g., reading audio data).
*   **`torch`:** The PyTorch library for working with tensors and neural networks.
*   **`warnings`:**  Allows us to suppress specific warning messages.
*   **`openvoice`:** The OpenVoice library (imported for `se_extractor` and `ToneColorConverter`).
*   **`melo.api`:**  The MeloTTS library (imported for the `TTS` class).
*   **`pydub`:**  For working with audio files.
*   **`streamlit`:** The Streamlit library for creating interactive web applications.
*   **`os`:**  Provides tools for interacting with the operating system (e.g., file paths).
*   **`tempfile`:**  For creating temporary files.

**3.2  OpenVoice Setup**

```python
# --- OpenVoice Setup ---
ckpt_converter = os.path.join(os.path.expanduser('~'), 'AudioGenerator', 'checkpoint', 'converter')
device = "cuda" if torch.cuda.is_available() else "cpu"
```

*   **`ckpt_converter`:**  This variable stores the path to the directory where the OpenVoice tone color converter checkpoints are located. It uses `os.path.expanduser('~')` to get the user's home directory and then appends the path to the checkpoints.
*   **`device`:**  Determines whether to use a CUDA-enabled GPU or the CPU for computations. It sets the `device` to `"cuda"` if a GPU is available, otherwise, it defaults to `"cpu"`.

**3.3  Main Function (`main`)**

```python
def main():
    # --- Streamlit UI ---
    st.set_page_config(layout="wide", page_title="Text-to-Speech with Tone Transfer")
    st.title("Text-to-Speech with Tone Transfer")

    # Text Input
    text_input = st.text_area("Enter your text:", height=200)

    # Language Selection
    language = st.selectbox("Select Language:", ["EN_NEWEST", "ES", "FR", "ZH", "JP", "KR"])

    # Reference Audio Upload
    uploaded_file = st.file_uploader("Upload Reference Audio (wav or mp3):", type=["wav", "mp3"])

    # --- Audio Generation and Tone Transfer ---
    if st.button("Generate Speech"):
        with st.spinner("Generating audio..."):
            if uploaded_file is not None:
                if not generate_speech(text_input, language, uploaded_file):
                    return
            else:
                st.warning("Please upload a reference audio file.")
```

*   **`st.set_page_config(layout="wide", page_title="Text-to-Speech with Tone Transfer")`:**  Configures the Streamlit page with a wide layout and sets the page title.
*   **`st.title("Text-to-Speech with Tone Transfer")`:**  Sets the main title for the Streamlit app.
*   **`text_input = st.text_area("Enter your text:", height=200)`:**  Creates a text area where users can input the text they want to convert to speech.
*   **`language = st.selectbox("Select Language:", ["EN_NEWEST", "ES", "FR", "ZH", "JP", "KR"])`:**  Creates a dropdown menu for users to choose the desired language for the speech synthesis.
*   **`uploaded_file = st.file_uploader("Upload Reference Audio (wav or mp3):", type=["wav", "mp3"])`:**  Creates a file uploader where users can upload a reference audio file (WAV or MP3) that will be used to transfer the speaker's tone.
*   **`if st.button("Generate Speech"):`:**  Creates a button labeled "Generate Speech." When clicked, the code within this block will be executed. 
    *   **`with st.spinner("Generating audio..."):`:**  Displays a spinner while the audio generation is in progress.
    *   **`if uploaded_file is not None:`:** Checks if a reference audio file has been uploaded.
    *   **`if not generate_speech(text_input, language, uploaded_file):`:** Calls the `generate_speech` function to handle the audio generation and tone transfer.
    *   **`st.warning("Please upload a reference audio file.")`:**  Displays a warning message if no reference audio file is uploaded.

**3.4  `generate_speech` Function**

```python
def generate_speech(text_input, language, uploaded_file):
    try:
        # Load the tone color converter
        tone_color_converter = ToneColorConverter(os.path.join(ckpt_converter, 'config.json'), device=device)
        tone_color_converter.load_ckpt(os.path.join(ckpt_converter, 'checkpoint.pth'))
    except FileNotFoundError:
        st.error("Error: 'config.json' or 'checkpoint.pth' not found. Please make sure the files exist in the correct location.")
        return False

    try:
        # Read uploaded audio file
        audio_data = uploaded_file.read()
        audio_format = uploaded_file.type.split('/')[1]

        # Convert MP3 to WAV if necessary
        if audio_format == 'mp3':
            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
            with NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
                audio_segment.export(temp_wav_file.name, format='wav')
                temp_wav_file_name = temp_wav_file.name
        else:
            with NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
                temp_wav_file.write(audio_data)
                temp_wav_file_name = temp_wav_file.name

        # Extract target speaker embedding from reference audio
        target_se, audio_name = se_extractor.get_se(temp_wav_file_name, tone_color_converter, vad=False)
    except Exception as e:
        st.error(f"Error extracting speaker embedding from reference audio: {e}")
        return False

    try:
        model = TTS(language=language, device=device)
        speaker_ids = model.hps.data.spk2id
        audio_segments = []

        for speaker_key in speaker_ids.keys():
            speaker_id = speaker_ids[speaker_key]
            speaker_key = speaker_key.lower().replace('_', '-')

            source_se = torch.load(os.path.join(os.path.expanduser('~'), 'AudioGenerator', 'checkpoint', 'base_speakers', 'ses', f'{speaker_key}.pth'), map_location=device)

            # Split text into sentences
            sentences = text_input.split('.')

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                try:
                    # Generate audio in memory (using io.BytesIO)
                    with NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
                        model.tts_to_file(sentence, speaker_id, temp_audio_file.name, speed=1.0)

                        audio_segment = AudioSegment.from_wav(temp_audio_file.name)

                        # Apply tone color conversion directly
                        with NamedTemporaryFile(delete=False, suffix='.wav') as output_temp_file:
                            tone_color_converter.convert(
                                audio_src_path=temp_audio_file.name,
                                src_se=source_se,
                                tgt_se=target_se,
                                output_path=output_temp_file.name,
                                message="@MyShell"
                            )

                            # Replace the original audio segment with the tone-converted one
                            audio_segment = AudioSegment.from_wav(output_temp_file.name)
                            audio_segments.append(audio_segment)

                except Exception as e:
                    st.error(f"Error during processing: {e}")
                    st.error("Trying to recover...")

            if audio_segments:
                # Merge audio segments for the current speaker
                merged_audio = audio_segments[0]
                for i in range(1, len(audio_segments)):
                    merged_audio += audio_segments[i]

                # Save the merged audio to a NamedTemporaryFile
                with NamedTemporaryFile(delete=False, suffix='.wav') as output_temp_file:
                    merged_audio.export(output_temp_file.name, format="wav")

                    # Download the audio in Streamlit
                    st.download_button(
                        label="Download Audio",
                        data=open(output_temp_file.name, 'rb'),
                        file_name=f"output_v2_{speaker_key}.wav",
                        mime="audio/wav"
                    )

                # Reset audio segments for the next speaker
                audio_segments = []

        st.success("Speech generation complete!")
        return True
    except Exception as e:
        st.error(f"Error during processing: {e}")
        st.error("Trying to recover...")
        return False

if __name__ == "__main__":
    main()
```

*   **`try-except` blocks:**  Error handling is used to catch potential exceptions during the process.
*   **`tone_color_converter = ToneColorConverter(os.path.join(ckpt_converter, 'config.json'), device=device)`:**  Creates an instance of the `ToneColorConverter` class from OpenVoice. This class is responsible for transferring the tone and speaker characteristics of the reference audio to the generated speech.
*   **`tone_color_converter.load_ckpt(os.path.join(ckpt_converter, 'checkpoint.pth'))`:**  Loads the tone color converter checkpoint, which contains the pre-trained model weights.
*   **Reading Audio File:**  Reads the uploaded audio file and determines its format.
*   **MP3 to WAV Conversion:**  Converts the audio file to WAV format if necessary.
*   **Extracting Speaker Embedding:**  Extracts the speaker embedding (a representation of the speaker's voice) from the reference audio file using the `se_extractor.get_se` function. The embedding will be used to transfer the speaker's tone to the generated speech.
*   **Initializing MeloTTS Model:**  Creates an instance of the `TTS` class from MeloTTS, setting the desired language and device.
*   **Looping Through Speakers:** Iterates through the available speakers in the MeloTTS model. 
    *   **`speaker_key`:** The key representing a specific speaker.
    *   **`speaker_id`:** The unique identifier of the speaker.
    *   **`speaker_key = speaker_key.lower().replace('_', '-')`:**  Prepares the speaker key for consistent naming.
    *   **`source_se = torch.load(os.path.join(os.path.expanduser('~'), 'AudioGenerator', 'checkpoint', 'base_speakers', 'ses', f'{speaker_key}.pth'), map_location=device)`:**  Loads the speaker embedding of the current speaker from a file, which contains the pre-trained embedding of the voice. 
    *   **Splitting Text into Sentences:** Splits the input text into sentences.
    *   **Generating Speech for Sentences:** Iterates through each sentence, generating speech using the MeloTTS model (`model.tts_to_file`) and then applying tone transfer using `tone_color_converter.convert`.
        *   **`model.tts_to_file`:** Generates speech for the sentence using the specified speaker ID.
        *   **`tone_color_converter.convert`:**  Applies the tone color conversion to the generated speech, transferring the speaker's tone from the reference audio.
    *   **Merging Audio Segments:** Merges the audio segments for each speaker into a single audio file.
    *   **Downloading Audio:** Provides a button for users to download the generated audio. 

**4. Running the App**

1.  **Save the code** as `main.py`.
2.  **Open your terminal** and navigate to the directory where you saved the file.
3.  **Run the Streamlit app:** 
    ```bash
    streamlit run main.py
    ```

**5. Using the App**

1.  **Open your web browser** and visit the address shown in the terminal.
2.  **Enter the text** you want to convert to speech.
3.  **Select the desired language.**
4.  **Upload a reference audio file** (WAV or MP3) that contains the voice whose tone you want to transfer.
5.  **Click "Generate Speech."**
6.  The app will generate the speech and provide a button to download the audio file.

### Conclusion

You've successfully built a voice cloning app using OpenVoice, MeloTTS, and Streamlit! Now, you can experiment with different voice clones and explore the creative possibilities of this exciting technology.

**Next Steps:**

*   **Experiment with Different Voices:**  Try cloning the voices of various speakers and see how the tone transfer works. 
*   **Fine-Tuning:**  Explore how to fine-tune the OpenVoice model for better voice quality and tone accuracy.
*   **Advanced Features:**  Consider adding features like speech synthesis with different emotions, voice modulation, or integration with other audio editing tools.
*   **Deployment:** Deploy your voice cloning app to a web server (like Heroku or AWS) to make it accessible online.

I hope this in-depth guide has been helpful in your journey into the world of voice cloning!  Feel free to experiment and have fun with this powerful technology. 
---

