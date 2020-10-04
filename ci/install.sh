#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

set -e

# Fix the compilers to workaround avoid having the Python 3.4 build
# lookup for g++44 unexpectedly.
export CC=gcc
export CXX=g++


if [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda-3.6.0-Linux-x86_64.sh \
        -O miniconda.sh
    chmod +x miniconda.sh && ./miniconda.sh -b
    export PATH=/home/travis/miniconda/bin:$PATH
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --yes python=$PYTHON_VERSION pip3 nose \
        numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION \
		nltk=3.0.2 scikit-learn=0.16.1 toolz=0.7.2 
    source activate testenv

    if [[ "$INSTALL_MKL" == "true" ]]; then
        # Make sure that MKL is used
        conda install --yes mkl
    else
        # Make sure that MKL is not used
        conda remove --yes --features mkl || echo "MKL not installed"
    fi

elif [[ "$DISTRIB" == "ubuntu" ]]; then
    # At the time of writing numpy 1.9.1 is included in the travis
    # virtualenv but we want to used numpy installed through apt-get
    # install.
    deactivate
    # Create a new virtualenv using system site packages for numpy and scipy
    virtualenv --system-site-packages testvenv
    source testvenv/bin/activate
    pip3 install nose
fi


pip3 install coverage coveralls lda


python3 -c "import nltk; nltk.download('maxent_treebank_pos_tagger'); nltk.download('punkt')"
# Build scikit-learn in the install.sh script to collapse the verbose
# build output in the travis output when it succeeds.
python3 --version
python3 -c "import numpy; print('numpy %s' % numpy.__version__)"
python3 -c "import scipy; print('scipy %s' % scipy.__version__)"
# python3 setup.py build_ext --inplace
