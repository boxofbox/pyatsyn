Changelog
=========

[v1.1.0](https://github.com/boxofbox/pyatsyn/releases/tag/v1.1.0) Variable Frame Rates, Transformation Tools, Misc Speed Ups (2023-??-??)
------------------------------------------
* NOTE: API breaking change! Folder and module naming refactoring to adopt more declarative structure -> analysis-transformation-synthesis
* variable frame rate structure ATSSoundVFR (ATSSound now inherits from this as it can be perceived as VFR with a consistent variation of 0)
* removed the optimized attribute from ATSSound* classes
* ATSSound*.optimize() sorting bug fix
* ATSSoundVFR -> ATSSound (CFR) conversion
* i/o updated to account for variable frame rates (adopting .atsv as file extension to distinguish it from .ats, since vfr is not externally supported yet)
* synth support for variable frame rates
* Various documentation updates
* Cross-synthesis capability added with pyatsyn.analysis.merge
* synthesis now allows an overriding sampling rate to be specified


[v1.0.0](https://github.com/boxofbox/pyatsyn/releases/tag/v1.0.0) Initial Release (2023-01-22)
-----------------------------------