import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import speech_recognition as sr
import soundfile as sf

from scipy.ndimage import gaussian_filter1d

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Speech Recognition Project",
    page_icon="🎤",
    layout="wide"
)

# ==========================================
# CUSTOM STYLE
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1 {
    color: #00FFD1;
    text-align: center;
    font-size: 55px;
}

h2, h3 {
    color: white;
}

.stButton>button {
    background-color: #00FFD1;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🎤 Speech Recognition")

st.sidebar.markdown("---")

st.sidebar.write("Speech Processing System")

st.sidebar.markdown("### Features")

st.sidebar.write("1. Signal Processing")
st.sidebar.write("2. Gaussian Filter")
st.sidebar.write("3. Silence Removal")
st.sidebar.write("4. ZCR")
st.sidebar.write("5. MFCC Extraction")
st.sidebar.write("6. Spectrogram")
st.sidebar.write("7. Speech Recognition")

st.sidebar.markdown("---")

st.sidebar.write("Developed by Mina")

# ==========================================
# TITLE
# ==========================================

st.title("🎤 Speech Recognition Project")

st.write("Modern Speech Processing System")

# ==========================================
# FILE UPLOAD
# ==========================================

audio_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

# ==========================================
# PROCESS AUDIO
# ==========================================

if audio_file is not None:

    # --------------------------------------
    # AUDIO PLAYER
    # --------------------------------------

    st.subheader("🎧 Audio Player")

    st.audio(audio_file)

    # --------------------------------------
    # LOAD AUDIO
    # --------------------------------------

    with st.spinner("Loading Audio..."):

        signal, sr_rate = librosa.load(audio_file)

    st.success("Audio Loaded Successfully")

    # --------------------------------------
    # METRICS
    # --------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Sample Rate", sr_rate)

    with col2:
        st.metric("Signal Length", len(signal))

    # ======================================
    # PREPROCESSING
    # ======================================

    smoothed_signal = gaussian_filter1d(
        signal,
        sigma=2
    )

    trimmed_signal, index = librosa.effects.trim(
        smoothed_signal,
        top_db=20
    )

    # ======================================
    # TABS
    # ======================================

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Waveform",
            "✨ Filters",
            "🎼 Features",
            "🗣 Recognition"
        ]
    )

    # ======================================
    # TAB 1 : WAVEFORM
    # ======================================

    with tab1:

        st.subheader("Original Signal")

        fig1, ax1 = plt.subplots(figsize=(12,4))

        ax1.plot(signal)

        ax1.set_title("Audio Signal")

        st.pyplot(fig1)

    # ======================================
    # TAB 2 : FILTERS
    # ======================================

    with tab2:

        col3, col4 = st.columns(2)

        with col3:

            st.subheader("Gaussian Filter")

            fig2, ax2 = plt.subplots(figsize=(8,4))

            ax2.plot(signal,
                     label='Original')

            ax2.plot(smoothed_signal,
                     label='Smoothed')

            ax2.legend()

            st.pyplot(fig2)

        with col4:

            st.subheader("Silence Removal")

            fig3, ax3 = plt.subplots(figsize=(8,4))

            ax3.plot(trimmed_signal)

            ax3.set_title("After Silence Removal")

            st.pyplot(fig3)

    # ======================================
    # TAB 3 : FEATURES
    # ======================================

    with tab3:

        # -----------------------------
        # ZCR
        # -----------------------------

        zcr = librosa.feature.zero_crossing_rate(
            trimmed_signal
        )

        st.subheader("📊 Zero Crossing Rate")

        fig4, ax4 = plt.subplots(figsize=(12,4))

        ax4.plot(zcr[0])

        ax4.set_title("ZCR")

        st.pyplot(fig4)

        st.metric("Average ZCR", round(np.mean(zcr), 4))

        # -----------------------------
        # MFCC
        # -----------------------------

        mfccs = librosa.feature.mfcc(
            y=trimmed_signal,
            sr=sr_rate,
            n_mfcc=13
        )

        st.subheader("🎼 MFCC Features")

        fig5, ax5 = plt.subplots(figsize=(12,6))

        img = librosa.display.specshow(
            mfccs,
            x_axis='time',
            ax=ax5
        )

        fig5.colorbar(img, ax=ax5)

        ax5.set_title("MFCC")

        st.pyplot(fig5)

        # -----------------------------
        # Spectrogram
        # -----------------------------

        stft = librosa.stft(trimmed_signal)

        spectrogram = librosa.amplitude_to_db(
            np.abs(stft)
        )

        st.subheader("🌈 Spectrogram")

        fig6, ax6 = plt.subplots(figsize=(12,6))

        img2 = librosa.display.specshow(
            spectrogram,
            sr=sr_rate,
            x_axis='time',
            y_axis='hz',
            ax=ax6
        )

        fig6.colorbar(img2, ax=ax6)

        ax6.set_title("Spectrogram")

        st.pyplot(fig6)

        # ======================================
        # TAB 4 : SPEECH RECOGNITION
        # ======================================

        with tab4:

            st.subheader("🗣 Speech Recognition")

            # Save clean PCM WAV
            sf.write(
                "speech_audio.wav",
                signal,
                sr_rate,
                subtype='PCM_16'
            )

            recognizer = sr.Recognizer()

            try:

                with sr.AudioFile("speech_audio.wav") as source:

                    audio_data = recognizer.record(source)

                    text = recognizer.recognize_google(
                        audio_data,
                        language="en-US"
                    )

                st.success("Speech Recognized Successfully")

                st.write("### Recognized Text")

                st.info(text)

            except Exception as e:

                st.error(str(e))

        # -----------------------------
        # Download Button
        # -----------------------------

        with open("clean_audio.wav", "rb") as file:

            st.download_button(
                label="⬇ Download Clean Audio",
                data=file,
                file_name="clean_audio.wav"
            )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    <div style='text-align: center; font-size:18px;'>
        Made with ❤️ by 
        <a href="https://www.linkedin.com/in/minamamdouh-/" target="_blank">
            Mina Mamdouh
        </a>
    </div>
    """,
    unsafe_allow_html=True
)