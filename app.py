import os
import random
import streamlit as st
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from huggingface_hub import login
import matplotlib.pyplot as plt

# ────────────────────────────
# Streamlit page configuration
# ────────────────────────────
st.set_page_config(
    page_title="SampleGen",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────────────────────
# Ensure 'samples' folder exists and load any pre-existing .wav files
# ──────────────────────────────────────────────────────────────
samples_dir = "samples"
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

if "generated_samples" not in st.session_state:
    st.session_state.generated_samples = []

# 2. Create “samples” folder if needed, otherwise load existing WAVs
samples_dir = "samples"
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)
else:
    if not st.session_state.generated_samples:
        for fname in sorted(os.listdir(samples_dir)):
            if fname.lower().endswith(".wav"):
                filepath = os.path.join(samples_dir, fname)
                st.session_state.generated_samples.append({
                    "name": os.path.splitext(fname)[0],
                    "path": filepath
                })
# ──────────────────────────────────────────────────────────────
# Initialize session state for Hugging Face login & vibe selection
# ──────────────────────────────────────────────────────────────
if "hf_logged_in" not in st.session_state:
    st.session_state.hf_logged_in = False

# For vibe toggles: track selection order and per-vibe booleans
vibes = ["Analog Warmth", "Modern Punch", "Lo-Fi Dust", "Cinematic", "Experimental"]
if "vibe_order" not in st.session_state:
    st.session_state.vibe_order = []
for v in vibes:
    key = f"vibe_selected_{v}"
    if key not in st.session_state:
        st.session_state[key] = False
if "selected_vibes" not in st.session_state:
    st.session_state.selected_vibes = []

# ──────────────────────────────────────────────────────────────
# CSS for sleeker look (hide footer, style buttons, etc.)
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Hide Streamlit footer & menu */
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Generate button styling */
    .generate-btn {
        width: 100%;
        padding: 12px;
        background-color: #ff5722;
        color: white;
        text-align: center;
        font-size: 18px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
    }
    .generate-btn:disabled {
        background-color: #ddd;
        color: #888;
        cursor: not-allowed;
    }

    /* Control group labels */
    .control-label {
        font-weight: 600;
        margin-top: 12px;
        margin-bottom: 4px;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ───────────────────────────────────────────────────────
# Load model & config (cached) - HF login used later
# ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model, cfg = get_pretrained_model("stabilityai/stable-audio-open-small")
    return model, cfg

model, model_cfg = load_model()
sample_rate = model_cfg["sample_rate"]
full_window = model_cfg["sample_size"]  # ~524288 samples (≈11.9 s)
hop_size = 1024

# ────────────────────────────────────────────────────────
# Sidebar: Hugging Face Token input
# ────────────────────────────────────────────────────────
with st.expander("API Settings (click to expand)", expanded=False):
    hf_token_input = st.text_input(
        "YOUR TOKEN HERE", 
        type="password", 
        help="Enter your Hugging Face API token to authenticate.",
    )
    if hf_token_input and not st.session_state.hf_logged_in:
        try:
            login(hf_token_input)
            st.session_state.hf_logged_in = True
            st.success("Logged in to Hugging Face successfully!")
        except Exception as e:
            st.error(f"Login failed: {e}")

# ─────────────────────────────────────────────────────────────
# Main layout: two columns – Left (controls), Right (preview & history)
# ─────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown("<span class='control-label'>1. Text Prompt</span>", unsafe_allow_html=True)
    prompt = st.text_area(
        label="Describe your sample (e.g. “Deep House bassline with rolling hi-hats”)", 
        height=80
    )

    # Vibe Buttons (as checkboxes with selection-limit logic)
    st.markdown("<span class='control-label'>2. Vibe (select up to 2)</span>", unsafe_allow_html=True)
    # Render checkboxes for each vibe
    for v in vibes:
        checkbox_key = f"vibe_selected_{v}"
        prev_state = st.session_state[checkbox_key]
        new_state = st.checkbox(v, value=prev_state, key=f"cb_{v}")
        # If changed, update order and enforce max of two
        if new_state != prev_state:
            st.session_state[checkbox_key] = new_state
            if new_state:
                # Added a vibe
                st.session_state.vibe_order.append(v)
            else:
                # Removed a vibe
                if v in st.session_state.vibe_order:
                    st.session_state.vibe_order.remove(v)
            # Enforce max 2: if >2, drop earliest selections
            while len(st.session_state.vibe_order) > 2:
                removed = st.session_state.vibe_order.pop(0)
                st.session_state[f"vibe_selected_{removed}"] = False

    # Update selected_vibes list
    st.session_state.selected_vibes = st.session_state.vibe_order.copy()

    # Tempo Slider
    st.markdown("<span class='control-label'>3. Tempo (BPM)</span>", unsafe_allow_html=True)
    tempo = st.slider("Tempo (BPM)", min_value=40, max_value=200, value=120, step=1)

    # Key & Scale
    st.markdown("<span class='control-label'>4. Key & Scale</span>", unsafe_allow_html=True)
    cols = st.columns(2)
    with cols[0]:
        key = st.selectbox("Key", ["C", "C♯/D♭", "D", "D♯/E♭", "E", "F", "F♯/G♭", "G", "G♯/A♭", "A", "A♯/B♭", "B"])
    with cols[1]:
        scale = st.selectbox("Scale", ["Major", "Minor", "Dorian", "Phrygian"])

    # Weirdness Knob (slider stand-in)
    st.markdown("<span class='control-label'>5. Weirdness (Temperature)</span>", unsafe_allow_html=True)
    weirdness = st.slider("Weirdness", min_value=0.0, max_value=2.0, value=1.0, step=0.01)

    # Density Knob
    st.markdown("<span class='control-label'>6. Rhythmic Density</span>", unsafe_allow_html=True)
    density = st.slider("Density", min_value=0.1, max_value=5.0, value=2.5, step=0.1)

    # Stereo Width Slider
    st.markdown("<span class='control-label'>7. Stereo Width</span>", unsafe_allow_html=True)
    stereo_width = st.slider(
        "Stereo Width",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="0.0 = Mono, 1.0 = Ultra-wide"
    )

    # One-Shot / Loop Toggle
    st.markdown("<span class='control-label'>8. Mode</span>", unsafe_allow_html=True)
    mode = st.radio("Choose:", ["One-Shot", "Loop"], index=0, horizontal=True)

    # Loop Length (active only if Loop)
    st.markdown("<span class='control-label'>9. Loop Length (seconds)</span>", unsafe_allow_html=True)
    if mode == "Loop":
        loop_length = st.number_input(
            "Loop Length (s)", 
            min_value=0.1, 
            max_value=float(full_window) / sample_rate,
            value=2.0, 
            step=0.1,
            help="Exact duration for seamless looping"
        )
    else:
        loop_length = None
        st.number_input(
            "Loop Length (s)", 
            min_value=0.1, 
            max_value=float(full_window) / sample_rate,
            value=1.0, 
            step=0.1, 
            disabled=True
        )

    # Sample Length + Chain Toggle
    st.markdown("<span class='control-label'>10. Sample Length (seconds)</span>", unsafe_allow_html=True)
    if mode == "Loop":
        sample_length = loop_length
        st.number_input("Sample Length (s)", value=sample_length, disabled=True, help="Matches Loop Length in Loop mode")
        chain_linked = True
    else:
        chain_linked = st.checkbox("Crop to this length", value=True, help="If unchecked, exports full internal window (~11s)")
        sample_length = st.number_input(
            "Sample Length (s)", 
            min_value=0.1, 
            max_value=float(full_window) / sample_rate,
            value=4.0, 
            step=0.1, 
            disabled=(not chain_linked)
        )

    # Advanced Options
    with st.expander("Advanced Settings"):
        # Noise Profile
        noise_profile = st.selectbox("Noise Profile", ["Pink", "White", "Brown"], index=1)
        # Seed Value
        seed_cols = st.columns([3, 1])
        with seed_cols[0]:
            seed_val = st.number_input("Seed Value", min_value=0, max_value=2**31-1, value=42, step=1)
        with seed_cols[1]:
            if st.button("🎲", help="Generate random seed"):
                seed_val = random.randint(0, 2**31-1)
                st.session_state.seed_val = seed_val
                st.experimental_rerun()

        # Style Transfer
        style_transfer = st.checkbox("Enable Style Transfer (upload sample)")
        if style_transfer:
            uploaded_file = st.file_uploader("Upload WAV/MP3", type=["wav", "mp3"])

# ─────────────────────────────────────────────────────────────
# Generate Button (bottom of left panel)
# ─────────────────────────────────────────────────────────────
with left_col:
    gen_button = st.button(
        "Generate Sample",
        key="generate",
        help="Click to run the diffusion model and create your sample",
        args=()
    )

# ─────────────────────────────────────────────────────────────
# Right Column: Preview & History
# ─────────────────────────────────────────────────────────────
with right_col:
    st.markdown("<span class='control-label'>▶️ Preview</span>", unsafe_allow_html=True)
    preview_container = st.empty()  # placeholder for waveform + player

    st.markdown("<span class='control-label'>🖼️ Recent Samples</span>", unsafe_allow_html=True)
    history_container = st.container()  # placeholder for gallery

# ─────────────────────────────────────────────────────────────
# Function: Plot waveform with matplotlib
# ─────────────────────────────────────────────────────────────
def plot_waveform(waveform, sr):
    fig, ax = plt.subplots(figsize=(8, 2))
    times = torch.linspace(0, waveform.shape[-1] / sr, waveform.shape[-1])
    ax.plot(times, waveform[0].numpy(), color="darkblue", linewidth=0.5)
    ax.fill_between(times, waveform[0].numpy(), color="lightblue", alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, waveform.shape[-1] / sr)
    ax.set_ylim(-1, 1)
    ax.set_title("Waveform Preview")
    ax.set_yticks([])
    st.pyplot(fig)

# ─────────────────────────────────────────────────────────────
# Function: Render Recent Samples Gallery
# ─────────────────────────────────────────────────────────────
def render_history():
    history = st.session_state.generated_samples
    if not history:
        st.info("No samples generated yet.")
        return

    # Iterate in groups of 3
    for i in range(0, len(history), 3):
        chunk = history[i : i + 3]                # e.g. [item0, item1, item2] or last row might be [itemN]
        cols = st.columns(len(chunk), gap="small")  # create exactly as many columns as items in this chunk

        for col, item in zip(cols, chunk):
            with col:
                # Sample name + audio player
                st.markdown(f"**{item['name']}**")
                st.audio(item["path"])

                # Three buttons in a single row
                btn_cols = st.columns([1, 1, 1], gap="small")
                play_btn, rename_btn, delete_btn = btn_cols

                if play_btn.button("🔊 Play", key=f"play_{item['path']}"):
                    # st.audio above will handle playback
                    pass

                if rename_btn.button("✎ Rename", key=f"rename_{item['path']}"):
                    new_name = st.text_input(
                        "New name:", 
                        value=item["name"], 
                        key=f"input_{item['path']}"
                    )
                    if new_name:
                        item["name"] = new_name
                        st.experimental_rerun()

                if delete_btn.button("🗑️ Delete", key=f"delete_{item['path']}"):
                    try:
                        os.remove(item["path"])
                    except FileNotFoundError:
                        pass
                    st.session_state.generated_samples.remove(item)
                    st.experimental_rerun()

# ─────────────────────────────────────────────────────────────
# Main: Handle Generate Button click
# ─────────────────────────────────────────────────────────────
if gen_button:
    if not st.session_state.hf_logged_in:
        st.error("Please enter a valid Hugging Face token in API Settings.")
    else:
        # Build conditioning dict
        conditioning = [{
            "prompt": prompt.strip(),
            "seconds_total": float(loop_length if mode == "Loop" else sample_length),
        }]

        # Always generate at full_window for best quality
        latent_window = full_window

        # Seed
        seed_val = st.session_state.get("seed_val", seed_val)
        torch.manual_seed(seed_val)

        with st.spinner("Generating… this may take a moment"):
            model = model.to("cpu")
            model.pretransform.model_half = False
            model.to(torch.float32)

            latent = generate_diffusion_cond(
                model,
                steps=8,
                cfg_scale=weirdness,
                conditioning=conditioning,
                sample_size=latent_window,
                sampler_type="pingpong",
                device="cpu",
            )

        # Rearrange and crop/pad
        audio = rearrange(latent, "b d n -> d (b n)").to(torch.float32)
        target_sec = loop_length if mode == "Loop" else sample_length
        target_samples = int(target_sec * sample_rate)
        if audio.shape[-1] < target_samples:
            audio = torch.nn.functional.pad(audio, (0, target_samples - audio.shape[-1]))
        else:
            audio = audio[:, :target_samples]

        # Normalize & convert to int16
        audio = (audio / audio.abs().max()).clamp(-1, 1)
        audio = (audio * 32767).to(torch.int16).cpu()

        # Save to file
        idx = len(st.session_state.generated_samples) + 1
        filename = f"sample_{idx}.wav"
        filepath = os.path.join(samples_dir, filename)
        torchaudio.save(filepath, audio, sample_rate)

        # Add to history
        st.session_state.generated_samples.append({
            "name": f"Sample {idx}",
            "path": filepath
        })

        # Display in preview
        with preview_container:
            st.success(f"✅ Generated: {filename} ({(audio.shape[-1]/sample_rate):.2f}s)")
            plot_waveform(audio, sample_rate)
            st.audio(filepath)

        # Refresh history
        with history_container:
            render_history()

# ─────────────────────────────────────────────────────────────
# On initial load (or if not generating), render history
# ─────────────────────────────────────────────────────────────
if not gen_button:
    with history_container:
        render_history()
