"""
FieldSense AI v3.0 - Setup Configuration

Installation:
    pip install -e .

Build distribution:
    python setup.py sdist bdist_wheel
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / 'README.md'
long_description = readme_path.read_text() if readme_path.exists() else ''

setup(
    name='fieldsense-ai',
    version='3.0.0',
    description='Real-time sports analytics with physics-based counterfactuals and live video processing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='FieldSense Team',
    author_email='team@fieldsense.ai',
    url='https://github.com/baloyitd/fieldsense-ai',
    packages=find_packages(
        where='.', 
        exclude=[
            'tests',
            'tests.*',
            'ui',
            'ui.*',
            'src-tauri',
            'src-tauri.*',
        ]
    ),
    package_dir={'': '.'},
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.24.0',
        'opencv-python>=4.8.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'reportlab>=4.0.0',
        'torch>=2.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pyinstaller>=5.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
        ],
        'kaggle': [
            'kaggle>=1.5.0',
        ],
        'ncaa': [
            'pyarrow>=12.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'fieldsense=bin.ship:main',
            'fieldsense-cert=src.privacy.cert:one_click_export',
            'ncaa-pipeline=ncaa_data.cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='sports analytics machine-learning video-processing privacy-compliant',
    project_urls={
        'Bug Reports': 'https://github.com/baloyitd/fieldsense-ai/issues',
        'Source': 'https://github.com/baloyitd/fieldsense-ai',
        'Documentation': 'https://github.com/baloyitd/fieldsense-ai/blob/main/README.md',
    },
)
