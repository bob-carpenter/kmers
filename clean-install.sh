#!/usr/bin/env bash

# quietly reload modules
module -q purge
module -q load gcc python intel-oneapi-mkl texlive

# remove old build relics and venv
rm -rf kmer-env _skbuild

# create and activate new venv
python -m venv kmer-env
source kmer-env/bin/activate

# build the shared object and install deps
pip install .

# reinstall in editable mode using the shared object artifact from the old install
pip install -e .
ln -s $(realpath _skbuild/*/cmake-install/lib/libkmers.so) kmer-env/lib

# if the C++ code ever changes, run the following commands to rebuild it
# module -q load gcc python intel-oneapi-mkl texlive
# source kmer-env/bin/activate
# pip install .
# pip install -e .
