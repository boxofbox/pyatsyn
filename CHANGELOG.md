Changelog
=========

v1.1.0 ADDED FEATURES TRACKER (2023-??-??)
------------------------------------------
* variable frame rate structure ATSSoundVFR (ATSSound now inherits from this as it can be perceived as VFR with a consistent variation of 0)
* removed the optimized attribute from ATSSound* classes
* ATSSound*.optimize() sorting bug fix
* ATSSoundVFR -> ATSSound (CFR) conversion
* NOTE: API breaking change! Folder and module naming refactoring to adopt more declarative structure -> analysis-transformation-synthesis
* i/o updated to account for variable frame rates (adopting .atsv as file extension to distinguish it from .ats, since vfr is not externally supported yet)


[https://github.com/boxofbox/pyatsyn/releases/tag/v1.0.0](v1.0.0) INITIAL RELEASE (2023-01-22)
-----------------------------------