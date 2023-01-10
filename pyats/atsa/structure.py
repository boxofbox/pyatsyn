from numpy import zeros, inf

class ats_peak:

    def __init__ (self, amp=0.0, frq=0.0, pha=0.0, smr=0.0, track=0, db_spl=0.0, 
                  barkfrq=0.0, slope_r=0.0, asleep_for=None, duration=1):
        self.amp = amp
        self.frq = frq
        self.pha = pha
        self.smr = smr        
        self.track = track
        self.db_spl = db_spl
        self.barkfrq = barkfrq
        self.slope_r = slope_r
        self.asleep_for = asleep_for
        self.duration = duration
        self.frq_max = 0.0
        self.amp_max = 0.0    
        self.frq_min = inf

    def clone (self):
        return ats_peak(self.amp,self.frq,self.pha,self.smr,self.track,self.db_spl,
                        self.barkfrq,self.slope_r,self.asleep_for, self.duration)

    def __repr__(self):
        return f"PK: f_{self.frq} at mag_{self.amp} + {self.pha}"

class ats_sound:
    """
    main data abstraction
    amp, frq, and pha contain sinusoidal modeling information as arrays of
    arrays of data arranged by partial par-energy and band-energy hold
    noise modeling information (experimental format)
    """    
    def __init__ (self, name, sampling_rate, frame_size, window_size, 
                  partials, frames, bands, dur, has_phase=True):
        self.name = name
        self.sampling_rate = sampling_rate
        self.frame_size = frame_size
        self.window_size = window_size
        self.partials = partials
        self.frames = frames
        self.bands = bands
        self.dur = dur
        # Info deduced from analysis
        self.optimized = False
        self.amp_max = 0.0
        self.frq_max = 0.0
        self.frq_av = zeros(partials,"float64")
        self.amp_av = zeros(partials,"float64")
        self.time = zeros(frames,"float64")
        self.frq = zeros([partials,frames],"float64")
        self.amp = zeros([partials,frames],"float64")
        self.pha = None
        if (has_phase):
            self.pha = zeros([partials,frames],"float64")
        # Noise Data
        self.energy = []        
        self.band_energy = []
        