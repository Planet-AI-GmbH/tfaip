Installation
============

|tfaip| is available on `pypi <https://pypi.org/project/tfaip>`_ and can thus be installed with `pip`:

.. code-block:: shell

    pip install tfaip


Prerequisites
-------------

You need to install the following on your system:

* Python 3.7 or later, including
* the python development packages (on Ubuntu ``apt install python3.7 python3.7-dev``, if not available in your distribution, use the `deadsnakes repo <https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa>`_ for ubuntu based distros)
* (optional) cuda/cudnn libs for GPU support, see `tensorflow <https://www.tensorflow.org/install/source#tested_build_configurations>`_ for the versions which are required/compatible.

Setup in a Virtual Environment
------------------------------

Setup your venv and install:

``<python>`` must be replaced with ``python3.7`` or later (check ``<python> --version`` before):

.. code-block:: shell

    virtualenv -p <python> PATH_TO_VENV
    source PATH_TO_VENV/bin/activate
    pip install -U pip  # recommended to get the latest version of pip
    pip install tfaip


Possible bugs
~~~~~~~~~~~~~

pycocotools
"""""""""""

Currently as of `numpy` 1.20, pycocotools are falsely compiled, run

.. code-block:: shell

    pip uninstall -y pycocotools
    pip install --no-cache-dir pycocotools

to fix your venv by reinstalling and recompiling the pycocotools.

Tests
-----

|tfaip| provides a set of tests which are listed in the `test` directory using the `unittest` framework.

Run the tests
~~~~~~~~~~~~~

Call ``python -m unittest`` or ``pytest`` to run the tests from the command line.

In PyCharm:  Right-click on ``test`` select ``Run 'Unittests in test'``, or "Shift-Shift", write "run" and choose ``Run 'Unittests in test'``


CI/CD
~~~~~
All tests will automatically run for CI/CD.


Development Setup
-----------------

For development support, clone the code, install the requirements in a fresh virtual environment, and link the |tfaip| source to the virtualenv:

.. code-block:: shell

    git clone https://github.com/Planet-AI-GmbH/tfaip.git  # or git clone git@github.com:Planet-AI-GmbH/tfaip.git
    cd tfaip
    virtualenv -p python3 venv
    source venv/bin/activate
    pip install -U pip  # recommended to get the latest version of pip
    pip install -r requirements.txt
    python setup.py develop
