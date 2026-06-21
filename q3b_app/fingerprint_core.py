
import numpy as np
import librosa
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.ndimage import maximum_filter

SAMPLE_RATE = 22050


def compute_spectrogram(y, sr, nperseg=2048, noverlap=None):
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Sxx = scipy_spectrogram(y, fs=sr, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db


def extract_peaks(Sxx_db, amp_threshold_db=-40, neighborhood_size=(20, 20)):
    local_max = maximum_filter(Sxx_db, size=neighborhood_size) == Sxx_db
    above_threshold = Sxx_db > amp_threshold_db
    peak_mask = local_max & above_threshold
    freq_idxs, time_idxs = np.where(peak_mask)
    return list(zip(freq_idxs, time_idxs))


def peaks_to_singles(peaks, t_axis):
    singles = {}
    for (fi, ti) in peaks:
        t_val = t_axis[ti]
        singles.setdefault(fi, []).append(t_val)
    return singles


def peaks_to_pairs(peaks, t_axis, fanout=5, min_dt=0.0, max_dt=4.0):
    peaks_sorted = sorted(peaks, key=lambda p: p[1])
    n = len(peaks_sorted)
    pairs = {}

    for i in range(n):
        f1, t1_idx = peaks_sorted[i]
        t1 = t_axis[t1_idx]
        count = 0
        for j in range(i + 1, n):
            f2, t2_idx = peaks_sorted[j]
            t2 = t_axis[t2_idx]
            dt = t2 - t1
            if dt < min_dt:
                continue
            if dt > max_dt:
                break
            dt_bin = int(round(dt * 50))
            key = (f1, f2, dt_bin)
            pairs.setdefault(key, []).append(t1)
            count += 1
            if count >= fanout:
                break

    return pairs


def fingerprint_audio(y, sr=SAMPLE_RATE):
    f_axis, t_axis, Sxx_db = compute_spectrogram(y, sr)
    peaks = extract_peaks(Sxx_db)
    singles = peaks_to_singles(peaks, t_axis)
    pairs = peaks_to_pairs(peaks, t_axis)
    return f_axis, t_axis, Sxx_db, peaks, singles, pairs


def match_using_pairs(pairs_q, pairs_db, bin_width=0.5):
    """Returns dict: song_name -> {offset_bin: vote_count}."""
    histograms = {}
    for key, query_times in pairs_q.items():
        if key not in pairs_db:
            continue
        db_entries = pairs_db[key]
        for q_time in query_times:
            for song_name, db_time in db_entries:
                offset = db_time - q_time
                offset_bin = round(offset / bin_width) * bin_width
                histograms.setdefault(song_name, {})
                histograms[song_name][offset_bin] = histograms[song_name].get(offset_bin, 0) + 1
    return histograms


def match_using_singles(singles_q, singles_db, bin_width=0.5):
    histograms = {}
    for freq_bin, query_times in singles_q.items():
        if freq_bin not in singles_db:
            continue
        db_entries = singles_db[freq_bin]
        for q_time in query_times:
            for song_name, db_time in db_entries:
                offset = db_time - q_time
                offset_bin = round(offset / bin_width) * bin_width
                histograms.setdefault(song_name, {})
                histograms[song_name][offset_bin] = histograms[song_name].get(offset_bin, 0) + 1
    return histograms


def best_match(histograms):
    if not histograms:
        return None, 0, 0

    song_best_peaks = {song: max(hist.values()) for song, hist in histograms.items()}
    ranked = sorted(song_best_peaks.items(), key=lambda x: x[1], reverse=True)
    best_song, best_count = ranked[0]
    second_count = ranked[1][1] if len(ranked) > 1 else 0
    return best_song, best_count, best_count - second_count


def identify_clip(y, sr, pairs_db, singles_db=None, use_pairs=True):
    f_axis, t_axis, Sxx_db, peaks, singles, pairs = fingerprint_audio(y, sr)

    if use_pairs:
        histograms = match_using_pairs(pairs, pairs_db)
    else:
        histograms = match_using_singles(singles, singles_db)

    pred_song, votes, margin = best_match(histograms)

    return {
        "f_axis": f_axis,
        "t_axis": t_axis,
        "Sxx_db": Sxx_db,
        "peaks": peaks,
        "predicted_song": pred_song,
        "votes": votes,
        "margin": margin,
        "histograms": histograms,
    }
