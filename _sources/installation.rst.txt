##############
Installation
##############

We have tried to precompile the binaries for as many platforms as
possible. If you are running one of the supported operating systems (which you most likely are), you can
install QArray via pip:

.. code:: bash

   pip install qarray

If you slip through the gaps, then the pip install will try to compile
the binaries for you. This might require you to install some additional
dependencies. In particular, you will need to have cmake and rust
installed.

Install Rust from: https://www.rust-lang.org/tools/install

Install CMake from: https://cmake.org/download/. On macOS and
Ubuntu, it may be more straightforward to install cmake using homebrew and apt, respectively.

Also, setting up JAX on macOS running on M series chips can be a bit
finicky. We outline the steps that worked for us in `macOS
installation <#macOS-installation>`__. Alternatively, just spin up a
`Github Codespace <https://github.com/codespaces>`__ and
``pip install qarray`` for an easy route around the various issues.

macOS installation from scratch
------------------

Getting JAX to work macOS on M Series chips can be problematic. Here
are the steps we used to get everything working starting from a fresh OS
install.

1. Install homebrew from https://brew.sh and run through the install
   script.

2. Use homebrew to install miniconda:

.. code:: zsh

   brew install  miniconda

3. Use homebrew to install cmake:

.. code:: zsh

   brew install cmake

4. Create a new conda environment and install pip:

.. code:: zsh

   conda create -n qarray python=3.11
   conda activate qarray
   conda install pip

5. Install qarray using pip:

.. code:: zsh

   pip install qarray

This installation script has been demonstrated to work on macOS Ventura
13.4 and Sonoma 14.4. To install directly from the repository, use the
command:

.. code:: zsh

   pip install git+https://github.com/b-vanstraaten/qarray.git@main


Jaxlib GPU
------------------

By default we install the CPU version of JAX, with the dependency 'jax[cpu]>=0.2', this is done
for compatibility reasons. If you want to use the GPU version of JAX, you need to install it by running:

.. code:: zsh

    pip install -U "jax[cuda12]"

This will install the GPU version of JAX with CUDA 12 support. If you are using a different version of CUDA, you can specify it by changing the version number in the brackets.
For more instructions see here: https://jax.readthedocs.io/en/latest/installation.html
