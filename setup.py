import os
import sys
import codecs

from setuptools import setup


version = '0.0.1.1'

HERE = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(HERE, 'README.md'), encoding='utf-8') as f:
    readme = f.read()


setup_requires = ['pytest-runner'] if \
    {'pytest', 'test', 'ptr'}.intersection(sys.argv) else []

setup(
    name='pyLPD',
    version=version,
    author='Gustavo Wiederhecker',
    author_email='gsw@g.unicamp.br',
    license='MIT v1.0',
    url='https://github.com/gwiederhecker/pyLPD',
    description='Python module implemention various functions used at our lab.',
    long_description=readme,
    keywords='instrument control, data processing, simulation scripts',
    packages=['pyLPD'],
    package_dir={'pyLPD': 'pyLPD'},
    ext_modules=[],
    provides=['pyLPD'],
    install_requires=['numpy', 'pyvisa', 'pandas', 'peakutils','scipy'] + (['future']
                                  if sys.version_info.major < 3 else []),
    setup_requires=setup_requires,
    tests_require=[],
    platforms='OS Independent',
    classifiers=[
        'Topic :: Scientific/Engineering :: Interface Engine/Protocol Translator',
        'Intended Audience :: Education',
        'Topic :: Education :: Computer Aided Instruction (CAI)'
    ],
    zip_safe=False,
    include_package_data=True)
