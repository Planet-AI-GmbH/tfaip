Development and Contributions
=============================

We highly encourage users of |tfaip| to contribute scenarios but also welcome improvements and bug fixes of the core classes and the documentation!

Create `github issues <https://github.com/Planet-AI-GmbH/tfaip/issues>`_ to report bugs or if you seek for support.
Open a pull request if you want to contribute.

Pull Request Checklist
----------------------

Before sending pull requests, make sure you do the following:

* Read the present contributing guidelines.
* Read the `Code of Conduct <https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md>`_
* Check if your changes are consistent with the guidelines.
* Check if your changes are consistent with the Coding Style.
* Run the unit tests.

Guidelines
----------

* Include unit tests when you contribute new features, as they help to a) prove that your code works correctly, and b) guard against future breaking changes to lower the maintenance cost.
* Bug fixes also generally require unit tests, because the presence of bugs usually indicates insufficient test coverage.
* Keep API compatibility in mind when you change code in |tfaip|
* As every PR requires several CPU time of CI testing, we discourage submitting PRs to fix one typo, one warning, etc. We recommend fixing the same issue at the file level at least (e.g.: fix all typos in a file, fix all compiler warning in a file, etc.)
* Tests should follow the testing best practices guide.


Licence
-------

Include the license at the top of new files:

.. code-block:: python

    # Copyright 2021 The tfaip authors. All Rights Reserved.
    #
    # This file is part of tfaip.
    #
    # tfaip is free software: you can redistribute it and/or modify
    # it under the terms of the GNU General Public License as published by the
    # Free Software Foundation, either version 3 of the License, or (at your
    # option) any later version.
    #
    # tfaip is distributed in the hope that it will be useful, but
    # WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
    # or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
    # more details.
    #
    # You should have received a copy of the GNU General Public License along with
    # tfaip. If not, see http://www.gnu.org/licenses/.
    # ==============================================================================

Coding Style
------------

|tfaip| follows `Black Python style guide <https://black.readthedocs.io>`_ and `PEP 8 <https://pep8.org/>`_, in short:
* Naming of packages: lowercase, no separators, e.g., `ctctransformer`.
* Naming of files: snake case, e.g., `data_params`
* Naming of classes: camelcase, e.g., `DataParamsAtr`
* Naming of variables: snake case

To manually run ``black`` on all files call:

.. code-block:: shell

    black .

To automatically apply the black code style on all files before committing (pre-commit hook) run (once):

.. code-block:: shell

    pre-commit install

To apply all ``pre-commit`` hooks once call:

.. code-block:: shell

    pre-commit run --all-files

For integrating black in your IDE (e.g., pycharm) see `here <https://black.readthedocs.io/en/stable/integrations/editors.html>`_.

Unittests
---------

All unittests can be run by calling

.. code-block:: shell

    pytest

To run tests in parallel install ``pytest-xdist`` via ``pip install pytest-xdist`` and call

.. code-block:: shell

    pytest -n 8

to run 8 tests in parallel. Note this requires some considerable amount of RAM.

Contributing Scenarios
----------------------

We welcome users to share their scenarios that are implemented with |tfaip|.
This helps other users to become familiar with |tfaip| but also to see if there is an already existing solution for a certain problem.
Simply create an issue providing a link to your scenario and a brief explanation.
Similar to the |tfaip| examples, we expect your scenario to ship unittests and a documentation.
