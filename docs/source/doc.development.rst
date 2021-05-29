Development
===========

This page is targeting developers not users of |tfaip|.
Read the [guidelines](#guidelines) first.

Guidelines
----------

These guidelines shall be followed when using |tfaip|.

Conventions
~~~~~~~~~~~

|tfaip| follows `Black Python style guide <https://black.readthedocs.io>`_, in short:
* Naming of packages: lowercase, no separators, e.g., `ctctransformer`.
* Naming of files: snake case, e.g., `data_params`
* Naming of classes: camelcase, e.g., `DataParamsAtr`
* Naming of variables: snake case

To automatically apply the black code style on all files before committing (pre-commit hook) run:

.. code-block:: shell

    pre-commit install

For integrating black in your IDE (e.g., pycharm) see `here <https://black.readthedocs.io/en/stable/integrations/editors.html>`_.

Logging
~~~~~~~

Use the `logging` module of python, e.g. start of each file:
```python
import logging
logger = logging.getLogger(__name__)
```

The |tfaip| Project Structure
-----------------------------

|tfaip| follows the structure of a basic python package

* **test**: Location for tests of the base classes but also the scenarios
* **tfaip**: Location of the actual python files. The |tfaip| directory comprises all python code relevant for "shipping" (no tests, no benchmarks) and has the following sub-dirs:
    * **base**: location of all base modules: ``data``, ``lav``, ``model``, ``scenario``, ``trainer``
    * **scenario**: Location of all available scenarios
    * **scripts**: python scripts for command line usage of |tfaip|
    * **util**: general utility methods
* **requirements.txt**: Pip requirements of the projects
