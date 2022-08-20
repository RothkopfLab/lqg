import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lqg",
    version="0.1.4",
    author="Dominik Straub",
    author_email="dominik.straub@tu-darmstadt.de",
    description="(Inverse) optimal control for linear quadratic Gaussian systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dominikstrb/lqg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "numpy>=1.20.3",
        "scipy>=1.6.3",
        "matplotlib==3.2.2",
        "numpyro==0.9.2",
        "jax>=0.3.14",
        "arviz>=0.11.2",
    ],
)
