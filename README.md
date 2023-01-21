


# ABOUT pyats

pyats is a python implementation of the Analysis-Transformation-Synthesis (ATS) spectral modeling system.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!TODO


---

pyats Copyright (c) <2023>, Johnathan G Lyon
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

for Mac: ```$ brew install libsndfile```
for Debian: ```$ apt-get install libsndfile-dev```

install via the PyPi repository

```
$ pip install pyats
```

# included command line utilities

```
$ pyats-atsa --help```
$ pyats-synth --help
```

Example command line usage to generate an ats file with residual:

```
$ pyats-atsa example.wav example.ats -v -r example-residual.wav
```

Example to synthesize the result using a sine-generator bank:

```
$ pyats-synth example.ats synthesized.wav
```

Example to synthesize the result a sine-generator bank and w/ band-limited noise synthesis for the residual:

```
$ pyats-synth example.ats synthesized_w_noise.wav --noise 1.0
```

---

# for Developers, if using from source

```
$ git clone https://github.com/boxofbox/pyats
```

i recommend running in a virtual environment from within the project base directory

```
$ cd pyats
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
```

may need to be run as a package in development mode
(from within the outermost pyats directory containing the pyproject.toml file)

```
$ pip install -e .
```