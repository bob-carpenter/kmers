from skbuild import setup
from setuptools import find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
classes = """
    Development Status :: 3 - Alpha
    Programming Language :: Python :: 3
    License :: OSI Approved :: BSD License
    Operating System :: OS Independent
"""
classifiers = [s.strip() for s in classes.split('\n') if s]

setup(name='kmerexpr',
      version='0.0.1',
      description='A Python package for estimating isoform expression',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Bob Carpenter, Robert Gower, Robert Blackwell, and Brian Ward',
      author_email='bcarpenter@flatironinstitute.org',
      url='https://github.com/bob-carpenter/kmers',
      packages=find_packages(),
      install_requires=['numpy>=1.18',
                        'scipy>=1.5',
                        'sparse-dot-mkl',
                        'numba',
                        ],
      cmake_args=['-DCMAKE_BUILD_TYPE=Release'],
      classifiers=classifiers,
      )
