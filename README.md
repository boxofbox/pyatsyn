


# ABOUT pyatsyn

pyatsyn is a python implementation of the Analysis-Transformation-Synthesis (ATS) spectral modeling system.

Analysis, Transformation and Synthesis (ATS) is a spectral modeling system based on a sinusoidal plus critical-band noise decomposition. The system can be used to analyze recorded sounds, transform their spectrum using a wide variety of algorithms and resynthesize them both out of time and in real time.

[pyatsyn Documentation at readthedocs.io](https://pyatsyn.readthedocs.io/)

---

pyatsyn Copyright (c) <2023>, Johnathan G Lyon
All rights reserved.

Except where otherwise noted, ATSA and ATSH is Copyright (c) <2002-2004>
Oscar Pablo Di Liscia, Pete Moss, and Juan Pampin

This source code is licensed under the BSD-style license found in the
LICENSE.rst file in the root directory of this source tree. 

---

# INSTALLATION

requires python 3.6+
developed/tested on python 3.9.15 using M1 Mac running macOS 13.0.1

requires LIBSNDFILE

for Mac (assuming Homebrew is installed): ```$ brew install libsndfile```
for Debian: ```$ apt-get install libsndfile-dev```

install via the PyPi repository

```
$ pip install pyatsyn
```

# included command line utilities

```
$ pyatsyn-atsa --help
$ pyatsyn-info --help
$ pyatsyn-synth --help
```

Example command line usage to generate an ats file with residual:

```
$ pyatsyn-atsa example.wav example.ats -v -r example-residual.wav
```

Example to print information about a .ats file to stdout:

```
$ pyatsyn-info example.ats
```     

Example to synthesize the result using a sine-generator bank:

```
$ pyatsyn-synth example.ats synthesized.wav
```

Example to synthesize the result a sine-generator bank and w/ band-limited noise synthesis for the residual:

```
$ pyatsyn-synth example.ats synthesized_w_noise.wav --noise 1.0
```

---

# for Developers, if using from source

```
$ git clone https://github.com/boxofbox/pyatsyn
```

i recommend running in a virtual environment from within the project base directory

```
$ cd pyatsyn
$ python -m venv .venv
$ source .venv/bin/activate
```


python libraries required: 

```
$ pip install numpy
$ pip install soundfile
```

optional documentation generation requires:

```
$pip install sphinx
$pip install sphinx_rtd_theme
```

may need to be run as a package in development mode
(from within the outermost pyatsyn directory containing the pyproject.toml file)

```
$ pip install -e .
```

If you are a newcomer to ATS, we recommend you start by looking at pyatsn/atsa/tracker.py