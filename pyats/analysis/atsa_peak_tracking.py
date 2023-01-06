def update_tracks(tracks, track_length, frame_n, analysis_frames, beta = 0.0):
    """
    updates the list of current <tracks>
    we use <track_length> frames of memory to update average amp, frq, and smr of the tracks
    the function returns a list of tracks
    If <tracks> is nil, a copy of peaks from the previous frame is returned
    """
    if tracks:
        
        frames = min(frame_n, track_length)
        first_frame = frame_n - frames

        for tk in tracks:
            track = tk.track
            frq_acc = 0
            f = 0
            amp_acc = 0
            a = 0
            smr_acc = 0
            s = 0
            last_frq = 0
            last_amp = 0
            last_smr = 0
            
            for i in range(first_frame, frame_n):

                l_peaks = analysis_frames[i]
                peak = find_track_in_peaks(track, l_peaks)

                if peak is not None:
                    if peak.frq > 0.0:
                        last_frq = peak.frq
                        frq_acc += peak.frq
                        f += 1
                    if peak.amp > 0.0:
                        last_amp = peak.amp
                        amp_acc += peak.amp
                        a += 1
                    if peak.smr > 0.0:
                        last_smr = peak.smr
                        smr_acc += peak.smr
                        s += 1

            if f > 0:
                tk.frq = ((1 - beta) * (frq_acc / f)) + (beta * last_frq)
            if a > 0:
                tk.amp = ((1 - beta) * (amp_acc / a)) + (beta * last_amp)
            if f > 0:
                tk.smr = ((1 - beta) * (smr_acc / s)) + (beta * last_smr)

        return tracks    

    else:
        # otherwise copy the previous analysis frame's peaks
        return [pk.clone() for pk in analysis_frames[frame_n - 1]]
        


def find_track_in_peaks(track, peaks):
    for pk in peaks:
        if track == pk.track:
            return pk
    return None


def peak_tracking(tracks, peaks, frequency_deviation = 0.45, SMR_continuity = 0.0):
    """
    adaptation of the Gale-Shapley algorithm for stable matching of prior tracks and new peaks
    for matched peaks, track numbers are updated
    return value is a list of unmatched tracks and a list of unmatched peaks
    """
    return None, None