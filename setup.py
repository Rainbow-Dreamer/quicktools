from setuptools import setup, find_packages
from os import path

setup(
    name='quicktools',
    packages = ['quicktools'],
    version='0.15',
    license='AGPLv3',
    description=
    'This is a python functions tools module for enhancing python grammar structures and functions',
    author='Rainbow-Dreamer',
    author_email='1036889495@qq.com',
    url='https://github.com/Rainbow-Dreamer/quicktools',
    download_url=
    'https://github.com/Rainbow-Dreamer/quicktools/archive/0.15.tar.gz',
    keywords=['tools', 'mathematics', 'statistics'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    include_package_data=True)
