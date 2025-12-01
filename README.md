# deTACH - Detaching spatial Trascriptomics into Annotated Cells with High confidence

A pre-release of DeTACH is called pytacs and is also available in pypi. No major difference currently. 

```
MIT License

Copyright (c) 2025 X. Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

A tool for assembling sub-cellular spots in subcellular-resolution spatial
transcriptomics into pseudo-single-cell spots and cell-type mapping leveraging paired scRNA-seq,
without reliance on imaging information.


## Installation

### Executable

An executable is available in [Release](https://github.com/LIU-Xnd/pydetach/releases).

### PyPI module

It could be simply
installed by `pip install pydetach` (Python version: 3.12).

For conda users,

```Bash
conda create -n detach python=3.12 -y
conda activate detach
pip install pydetach
```

For python3 users, first make sure your python is
of version 3.12, and then in your working directory,

```Bash
python -m venv detach
source detach/bin/activate
python -m pip install pydetach
```

To use it for downstream analysis in combination with Squidpy, it is recommended to use a seperate virtual environment to install Squidpy.

## Usage

deTACH now is packed as a commandline module:

See help:

```
$ detach -h
```

if you've downloaded the executable, or

```
$ python -m pydetach -h
```

if you've installed the Python module.

If you are interested in using pydetach as a python package, we recommend using pydetach.recipe module.
See docstring therein.
