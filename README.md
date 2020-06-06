![PyLops-distributed](https://github.com/equinor/pylops-distributed/blob/master/docs/source/_static/distr-pylops_b.png)

[![PyPI version](https://badge.fury.io/py/pylops-distributed.svg)](https://badge.fury.io/py/pylops-distributed)
[![Build Status](https://travis-ci.org/equinor/pylops-distributed.svg?branch=master)](https://travis-ci.org/equinor/pylops-distributed)
[![AzureDevOps Status](https://dev.azure.com/matteoravasi/PyLops/_apis/build/status/equinor.pylops-distributed?branchName=master)](https://dev.azure.com/matteoravasi/PyLops/_build/latest?definitionId=4&branchName=master)
[![Documentation Status](https://readthedocs.org/projects/pylops-distributed/badge/?version=latest)](https://pylops-distributed.readthedocs.io/en/latest/?badge=latest)
[![OS-support](https://img.shields.io/badge/OS-linux,osx-850A8B.svg)](https://github.com/equinor/pylops)
[![Slack Status](https://img.shields.io/badge/chat-slack-green.svg)](https://pylops.slack.com)


:vertical_traffic_light: :vertical_traffic_light: This library is under early development.
Expect things to constantly change until version v1.0.0. :vertical_traffic_light: :vertical_traffic_light:

## Objective
This library is an extension of [PyLops](https://pylops.readthedocs.io/en/latest/)
for distributed operators.

As much as [numpy](http://www.numpy.org) and [scipy](http://www.scipy.org/scipylib/index.html) lie
at the core of the parent project PyLops, PyLops-distributed heavily builds on top of
[Dask](https://dask.org), a Python library for distributed computing.

Doing so, linear operators can be distributed across several processes on a single node
or even across multiple nodes. Their forward and adjoint
are first lazily built as directed acyclic graphs and evaluated only when requested by
the user (or automatically within one of our solvers).

Here is a simple example showing how a diagonal operator can be created,
applied and inverted using PyLops:
```python
import numpy as np
from pylops import Diagonal

n = 10
x = np.ones(n)
d = np.arange(n) + 1

Dop = Diagonal(d)

# y = Dx
y = Dop*x
# x = D'y
xadj = Dop.H*y
# xinv = D^-1 y
xinv = Dop / y
```

and similarly using PyLops-distributed:
```python
import numpy as np
import dask.array as da
import pylops_distributed
from pylops_distributed import Diagonal

# set-up client
client = pylops_distributed.utils.backend.dask()

n = 10
x = da.ones(n, chunks=(n//2,))
d = da.from_array(np.arange(n) + 1, chunks=(n//2, n//2))

Dop = Diagonal(d)

# y = Dx
y = Dop*x
# x = D'y
xadj = Dop.H*y
# xinv = D^-1 y
xinv = Dop / y

da.compute((y, xadj, xinv))
client.close()
```

It is worth noticing two things at this point:

- in this specific case we did not even need to reimplement the ``Derivative`` operator.
  Calling numpy operations as methods (e.g., ``x.sum()``) instead of functions (e.g., ``np.sum(x)``)
  makes it automatic for our operator to act as a distributed operator when a dask array is provided instead. Unfortunately not all numpy functions are also implemented as methods: in those cases we
  reimplement the operator directly within PyLops-distributed.
- Using ``*`` and ``.H*`` is still possible also within PyLops-distributed, however when initializing an
  operator we will need to decide whether we want to simply create dask graph or also evaluation.
  This gives flexibility as we can decide if and when apply evaluation using the ``compute`` method
  on a dask array of choice.


## Getting started

You need **Python 3.5 or greater**.

#### From PyPi
Coming soon...

#### From Github

You can also directly install from the master node

```
pip install https://git@github.com/equinor/pylops-distributed.git@master
```

## Contributing
*Feel like contributing to the project? Adding new operators or tutorial?*

Follow the instructions from [PyLops official documentation](https://pylops.readthedocs.io/en/latest/contributing.html).

## Documentation
The official documentation of PyLops-distributed is available [here](https://pylops-distributed.readthedocs.io/).


Moreover, if you have installed PyLops using the *developer environment* you can also build the documentation locally by
typing the following command:
```
make doc
```
Once the documentation is created, you can make any change to the source code and rebuild the documentation by
simply typing
```
make docupdate
```
Note that if a new example or tutorial is created (and if any change is made to a previously available example or tutorial)
you are required to rebuild the entire documentation before your changes will be visible.


## History
PyLops-Distributed was initially written and it is currently maintained by [Equinor](https://www.equinor.com).
It is an extension of [PyLops](https://pylops.readthedocs.io/en/latest/) for large-scale optimization with
*distributed* linear operators that can be tailored to our needs, and as contribution to the free software community.


## Contributors
* Matteo Ravasi, mrava87