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
import os

from setuptools import setup, find_packages

this_dir = os.path.dirname(os.path.realpath(__file__))

# Parse version
main_ns = {}
with open(os.path.join(this_dir, "tfaip", "version.py")) as f:
    exec(f.read(), main_ns)
    __version__ = main_ns["__version__"]

setup(
    name="tfaip",
    version=__version__,
    packages=find_packages(exclude=["test/*"]),
    license="GPL-v3.0",
    long_description=open(os.path.join(this_dir, "README.md")).read(),
    long_description_content_type="text/markdown",
    author="Planet AI GmbH",
    author_email="admin@planet-ai.de",
    url="https://github.com/Planet-AI-GmbH/tf2_aip_base",
    download_url="https://github.com/Planet-AI-GmbH/tf2_aip_base/archive/{}.tar.gz".format(__version__),
    entry_points={
        "console_scripts": [
            "tfaip-train=tfaip.scripts.train:run",
            "tfaip-lav=tfaip.scripts.lav:run",
            "tfaip-multi-lav=tfaip.scripts.lav_multi:run",
            "tfaip-evaluate=tfaip.scripts.evaluate:run",
            "tfaip-predict=tfaip.scripts.predict:run",
            "tfaip-experimenter=tfaip.scripts.experimenter:main",
            "tfaip-resume-training=tfaip.scripts.resume_training:main",
            "tfaip-train-from-params=tfaip.scripts.train_from_params:main",
        ],
    },
    python_requires=">=3.7",
    install_requires=open(os.path.join(this_dir, "requirements.txt")).read().split("\n"),
    keywords=["machine learning", "tensorflow", "framework"],
    data_files=[("", [os.path.join(this_dir, "requirements.txt")])],
)
