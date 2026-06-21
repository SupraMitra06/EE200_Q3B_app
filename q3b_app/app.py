
import os
import io
import pickle

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import librosa
import pandas as pd

import fingerprint_core as fc

st.set_page_config(page_title="Signal Lock — Audio Fingerprinting", layout="wide", page_icon="◈")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "fingerprint_db.pkl")
SAMPLE_RATE = fc.SAMPLE_RATE

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
INK ="#0B0E14"          # near-black background, like a spectrum analyzer
PANEL = "#11151F"        # slightly lifted panel background
LINE = "#1E2433"         # hairline borders
CYAN = "#22D3EE"         # primary signal color
MAGENTA = "#F472B6"      # secondary accent
AMBER = "#FBBF24"        # "match found" highlight
SLATE = "#94A3B8"        # secondary text
FOG = "#E2E8F0"          # primary text

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background-color: {INK};
    color: {FOG};
}}

section[data-testid="stSidebar"] {{
    background-color: {PANEL};
    border-right: 1px solid {LINE};
}}

.signal-hero {{
    border: 1px solid {LINE};
    background: linear-gradient(135deg, {PANEL} 0%, {INK} 100%);
    border-radius: 6px;
    padding: 28px 32px;
    margin-bottom: 28px;
}}
.signal-eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {CYAN};
    margin-bottom: 8px;
}}
.signal-title {{
    font-size: 30px;
    font-weight: 700;
    color: {FOG};
    margin: 0 0 6px 0;
}}
.signal-subtitle {{
    font-size: 14px;
    color: {SLATE};
    margin: 0;
}}

.waveform-strip {{
    display: flex;
    align-items: center;
    gap: 3px;
    height: 28px;
    margin-top: 18px;
}}
.waveform-strip span {{
    display: inline-block;
    width: 3px;
    background: {CYAN};
    border-radius: 2px;
    animation: pulse 1.6s ease-in-out infinite;
    opacity: 0.85;
}}
@keyframes pulse {{
    0%, 100% {{ height: 20%; }}
    50% {{ height: 100%; }}
}}

.stat-chip {{
    font-family: 'IBM Plex Mono', monospace;
    background: {PANEL};
    border: 1px solid {LINE};
    border-radius: 4px;
    padding: 10px 16px;
    display: inline-block;
    margin-right: 10px;
    font-size: 13px;
    color: {SLATE};
}}
.stat-chip b {{ color: {CYAN}; }}

.match-card {{
    border: 1px solid {AMBER};
    background: rgba(251, 191, 36, 0.06);
    border-radius: 6px;
    padding: 20px 24px;
    margin: 16px 0;
}}
.match-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {AMBER};
}}
.match-name {{
    font-size: 22px;
    font-weight: 700;
    color: {FOG};
    margin: 4px 0 10px 0;
}}
.match-meta {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: {SLATE};
}}
.match-meta b {{ color: {CYAN}; }}

.no-match-card {{
    border: 1px solid {MAGENTA};
    background: rgba(244, 114, 182, 0.06);
    border-radius: 6px;
    padding: 20px 24px;
    margin: 16px 0;
    color: {SLATE};
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
}}

.section-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {SLATE};
    border-bottom: 1px solid {LINE};
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}}

h1, h2, h3 {{ color: {FOG} !important; }}
hr {{ border-color: {LINE} !important; }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOT_STYLE = {
    "figure.facecolor": INK,
    "axes.facecolor": PANEL,
    "axes.edgecolor": LINE,
    "axes.labelcolor": FOG,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "text.color": FOG,
    "axes.titlecolor": FOG,
    "grid.color": LINE,
    "font.family": "monospace",
}
plt.rcParams.update(PLOT_STYLE)

SIGNAL_CMAP = LinearSegmentedColormap.from_list(
    "signal_lock", [INK, "#163447", "#1B6E8C", CYAN, "#FDE68A"]
)


@st.cache_resource
def load_database():
    """Loads the prebuilt hash database once and caches it across reruns."""
    if not os.path.exists(DB_PATH):
        st.error(
            f"Database file '{DB_PATH}' not found. Make sure it's in the same "
            f"folder as app.py before deploying."
        )
        st.stop()
    with open(DB_PATH, "rb") as f:
        data = pickle.load(f)
    return data["singles_db"], data["pairs_db"]


def load_audio_from_upload(uploaded_file, sr=SAMPLE_RATE):
    """Loads an uploaded audio file (mp3/wav) into a mono numpy array."""
    audio_bytes = uploaded_file.read()
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
    return y


def plot_spectrogram_with_peaks(f_axis, t_axis, Sxx_db, peaks, title="Spectrogram + Constellation"):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.pcolormesh(t_axis, f_axis, Sxx_db, shading="auto", cmap=SIGNAL_CMAP)
    if peaks:
        peak_times = [t_axis[ti] for (fi, ti) in peaks]
        peak_freqs = [f_axis[fi] for (fi, ti) in peaks]
        ax.scatter(peak_times, peak_freqs, s=14, c=MAGENTA, marker="o",
                   edgecolors="none", label="peaks", alpha=0.9)
        ax.legend(loc="upper right", facecolor=PANEL, edgecolor=LINE, labelcolor=FOG, fontsize=9)
    ax.set_ylim(0, 5000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    for spine in ax.spines.values():
        spine.set_color(LINE)
    fig.patch.set_facecolor(INK)
    fig.tight_layout()
    return fig


def plot_offset_histogram(histograms, song_name, title="Offset histogram"):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    hist = histograms.get(song_name, {})
    if hist:
        offsets = sorted(hist.keys())
        counts = [hist[o] for o in offsets]
        max_count = max(counts)
        colors = [AMBER if c == max_count else CYAN for c in counts]
        ax.bar(offsets, counts, width=0.4, color=colors)
    ax.set_xlabel("Offset (s)")
    ax.set_ylabel("Vote count")
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    for spine in ax.spines.values():
        spine.set_color(LINE)
    fig.patch.set_facecolor(INK)
    fig.tight_layout()
    return fig


def run_identification(y, sr, pairs_db, singles_db, use_pairs=True):
    return fc.identify_clip(y, sr, pairs_db, singles_db, use_pairs=use_pairs)


def waveform_html(n_bars=40):
    """Tiny ambient animated waveform strip for the header."""
    rng = np.random.default_rng(7)
    delays = rng.uniform(0, 1.6, n_bars)
    bars = "".join(
        f'<span style="animation-delay:{d:.2f}s"></span>' for d in delays
    )
    return f'<div class="waveform-strip">{bars}</div>'


st.markdown(
    f"""
    <div class="signal-hero">
        <div class="signal-eyebrow">EE200 · Course Project · Q3B</div>
        <div class="signal-title">◈ Signal Lock</div>
        <p class="signal-subtitle">A constellation-and-hash audio fingerprinting identifier — built from spectrogram peaks, paired into compact hashes, matched by offset-histogram voting.</p>
        {waveform_html()}
    </div>
    """,
    unsafe_allow_html=True,
)

singles_db, pairs_db = load_database()

with st.sidebar:
    st.markdown('<div class="section-label">Database</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="stat-chip">Paired hashes<br><b>{len(pairs_db):,}</b></div>
        <div class="stat-chip">Single-peak keys<br><b>{len(singles_db):,}</b></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">Mode</div>', unsafe_allow_html=True)
    mode = st.radio("Mode", ["Single-clip mode", "Batch mode"], label_visibility="collapsed")

if mode == "Single-clip mode":
    st.markdown('<div class="section-label">Query</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a query clip (mp3 or wav)", type=["mp3", "wav"])

    if uploaded is not None:
        with st.spinner("Listening... extracting peaks and matching hashes"):
            y = load_audio_from_upload(uploaded)
            result = run_identification(y, SAMPLE_RATE, pairs_db, singles_db, use_pairs=True)

        pred_song = result["predicted_song"]

        if pred_song is None:
            st.markdown(
                '<div class="no-match-card">◈ NO MATCH — query hashes did not '
                'align with any song in the database at a consistent offset.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="match-card">
                    <div class="match-label">Match found</div>
                    <div class="match-name">{pred_song}</div>
                    <div class="match-meta">votes <b>{result['votes']}</b> &nbsp;·&nbsp; margin over 2nd best <b>{result['margin']}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-label">Intermediate steps</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            fig1 = plot_spectrogram_with_peaks(
                result["f_axis"], result["t_axis"], result["Sxx_db"], result["peaks"],
                title="Spectrogram + Constellation"
            )
            st.pyplot(fig1)

        with col2:
            if pred_song is not None:
                fig2 = plot_offset_histogram(result["histograms"], pred_song,
                                              title=f"Offset histogram — {pred_song}")
                st.pyplot(fig2)
            else:
                st.markdown(
                    '<div class="no-match-card">No histogram to display — no match found.</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("Top 5 candidate songs by vote count"):
            ranked = sorted(
                ((song, max(hist.values())) for song, hist in result["histograms"].items()),
                key=lambda x: x[1], reverse=True
            )[:5]
            if ranked:
                st.dataframe(
                    pd.DataFrame(ranked, columns=["Song", "Peak votes"]),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.write("No candidates found.")


else:
    st.markdown('<div class="section-label">Batch query</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{SLATE}; font-size:14px;">Upload multiple query clips at once. '
        f'Produces <code>results.csv</code> with columns <code>filename,prediction</code> '
        f'(prediction = matched song\'s filename without extension).</p>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload query clips (mp3 or wav)", type=["mp3", "wav"], accept_multiple_files=True
    )

    if uploaded_files:
        if st.button(f"Run identification on {len(uploaded_files)} clips"):
            rows = []
            progress = st.progress(0)

            for i, uploaded in enumerate(uploaded_files):
                filename = uploaded.name
                try:
                    y = load_audio_from_upload(uploaded)
                    result = run_identification(y, SAMPLE_RATE, pairs_db, singles_db, use_pairs=True)
                    prediction = result["predicted_song"] if result["predicted_song"] else "no_match"
                except Exception as e:
                    prediction = "error"
                    st.warning(f"Failed to process {filename}: {e}")

                rows.append({"filename": filename, "prediction": prediction})
                progress.progress((i + 1) / len(uploaded_files))

            results_df = pd.DataFrame(rows, columns=["filename", "prediction"])
            st.markdown(
                '<div class="match-card"><div class="match-label">Batch complete</div></div>',
                unsafe_allow_html=True,
            )
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results.csv",
                data=csv_bytes,
                file_name="results.csv",
                mime="text/csv",
            )
