[PyLops-distributed](https://github.com/equinor/pylops-distributed/blob/master/docs/source/_static/d-pylops_b.png)

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
import pylops_distributed
from pylops import Diagonal

# set-up client
client = pylops_distributed.utils.backend.dask()
client

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

```
