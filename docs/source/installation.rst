Installation
===========

Requirements
-----------

prob-spaces requires the following packages:

* Python 3.10 or higher
* PyTorch
* NumPy
* Gymnasium
* TorchRL

From PyPI
---------

To install prob-spaces from PyPI:

.. code-block:: bash

    pip install prob-spaces

From Source
-----------

To install from source:

.. code-block:: bash

    git clone https://github.com/DanielAvdar/prob-spaces.git
    cd prob-spaces
    pip install -e .

Development Installation
-----------------------

For development installation:

.. code-block:: bash

    git clone https://github.com/DanielAvdar/prob-spaces.git
    cd prob-spaces
    pip install -e ".[dev]"

GPU Support
----------

prob-spaces uses PyTorch, which can be installed with CUDA support for GPU acceleration.
The package configuration includes a PyTorch CUDA 12.4 index. To use a different CUDA version,
you may need to modify the PyTorch installation separately.
