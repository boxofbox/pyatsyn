class ats_peak:

    def __init__ (self, amp=0.0, frq=0.0, pha=0.0, smr=0.0, track=0, db_spl=0.0, 
                  barkfrq=0.0, slope_r=0.0):
        self.amp = amp
        self.frq = frq
        self.pha = pha
        self.smr = smr        
        self.track = track
        self.db_spl = db_spl
        self.barkfrq = barkfrq
        self.slope_r = slope_r

    def clone (self):
        return ats_peak(self.amp,self.frq,self.pha,self.smr,self.track,self.db_spl,
                        self.barkfrq,self.slope_r)

    def __repr__(self):
        return f"PK: f_{self.frq} at mag_{self.amp} + {self.pha}"