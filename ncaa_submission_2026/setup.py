"""
setup.py — FieldSense NCAA 2026 Submission Package
"""

from setuptools import setup, find_packages

setup(
    name="fieldsense-ncaa-2026",
    version="2026.1.0",
    description="FieldSense NCAA 2026 tournament prediction system",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=2.0",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "scipy>=1.10",
        "pyarrow>=12.0",
    ],
    extras_require={
        "neural": ["torch>=2.0", "onnxruntime>=1.15"],
        "test": ["pytest>=7.0", "pytest-cov>=4.0"],
    },
    entry_points={
        "console_scripts": [
            "fieldsense-predict=ncaa_submission_2026.run_prediction:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
