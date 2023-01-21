pyats
=====

TODO INSTALLATION
 
for newcomers start with tracker

TODO ABOUT & copy to github readme

TODO Licence & copyright

.. toctree::
   :maxdepth: 2
   :caption: API

   pyats   

ATS Overview
============

Analysis, Transformation and Synthesis (ATS) is a spectral modeling system based on a 
sinusoidal plus critical-band noise decomposition. The system can be used to analyze 
recorded sounds, transform their spectrum using a wide variety of algorithms and 
resynthesize them both out of time and in real time.

.. image:: _static/img/ats_block.png
        :width: 350
        :alt: graphic depiction of smr calculation

Psychoacoustic processing informs the system's sinusoidal tracking and noise modeling 
algorithms. Perceptual Audio Coding (PAC) techniques such as Signal-to-Mask Ratio (SMR) 
evaluation are used to achieve perceptually accurate sinusoidal tracking. SMR values 
are also used as a psychoacoustic metric to determine the perceptual relevance of partials 
during analysis data postprocessing. The system's noise component is modeled using 
Bark-scale frequency warping and sub-band noise energy evaluation. Noise energy at the 
sub-bands is then distributed on a frame-by-frame basis among the partials resulting 
in a compact hybrid representation based on noise modulated sinusoidal trajectories.

Other ATS Implementations
=========================

Originally implemented in LISP, using the CLM sound synthesis and processing language, 
ATS has been ported to C in the form of a spectral modeling library. This library, 
called ATSA, implements the ATS system API which has served as foundation for the 
development of the ATSH graphic user interface. Written in GTK+, ATSH not only provides 
user- friendly access to the ATS analysis/synthesis core but also graphic data editing 
and transformation tools. ATS interfaces for SuperCollider, Csound and PD have also 
been developed.

https://gitlab.com/dxarts/projects/ats

SuperCollider
Interfaces for ATS (including classes to read ATS files as well ad UGens to do transformation and synthesis) are included in Josh Parmenter's UGen library, JoshUGens, distributed in the sc3-plugins package for SuperCollider.

https://supercollider.github.io/sc3-plugins/


http://flossmanual.csound.com/sound-modification/ats-resynthesis

PureData
Pablo Di Liscia's Pure Data Binaries and Toolkit
https://puredata.info/Members/pdiliscia/ats-pd


ATS File Format http://floss.booktype.pro/csound/k-ats-resynthesis/

ATS Theory
==========

`Link to pdf <_static/pdf/ats_theory.pdf>`_

ATS: A System for Sound Analysis Transformation and Synthesis Based on a Sinusoidal plus Critical-Band Noise Model and Psychoacoustics
Juan Pampin
Center for Digital Arts and Experimental Media (DXARTS), University of Washington pampin@u.washington.edu

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
