from setuptools import setup, find_packages

setup(  
        name='pyats', 
        version='1.0.0', 
        packages=find_packages(),
        entry_points={
            'console_scripts': [
                'pyats-atsa = pyats.atsa.tracker:tracker_CLI',
                'pyats-synth = pyats.ats_synth:synth_CLI',
            ]
        }    
        
    )

