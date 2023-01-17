requires python 3.8+
developed/tested on python 3.9.15 using M1 Mac running macOS 13.0.1

requires LIBSNDFILE

`brew install libsndfile`

# for Developers, if using from source
i recommend running in a virtual environment from within the project base directory

`python -m venv .venv`

`source .venv/bin/activate`


python libraries required: 

`pip install numpy`

`pip install soundfile`

may need to be run as a package in development mode
(from within the outermost pyats directory)

`pip install -e .`