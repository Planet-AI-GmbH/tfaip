from setuptools import setup, find_packages
from tfaip import __version__

setup(
    name='tfaip',
    version=__version__,
    packages=find_packages(),
    license='',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    author="Planet AI GmbH",
    author_email="admin@planet-ai.de",
    url="https://github.com/Planet-AI-GmbH/tf2_aip_base",
    download_url='https://github.com/Planet-AI-GmbH/tf2_aip_base/archive/{}.tar.gz'.format(__version__),
    entry_points={
        'console_scripts': [
            'tfaip-train=tfaip.scripts.train:run',
            'tfaip-lav=tfaip.scripts.lav:run',
            'tfaip-experimenter=tfaip.scripts.experimenter:main',
            'tfaip-resume-training=tfaip.scripts.resume_training:main',
        ],
    },
    python_requires='>=3.7',
    install_requires=open("requirements.txt").read().split('\n'),
    tests_requires=[],
    keywords=['machine learning', 'tensorflow'],
    data_files=[('', ["requirements.txt"])],
)
