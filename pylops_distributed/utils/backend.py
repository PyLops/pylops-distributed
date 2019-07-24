from dask.distributed import Client, LocalCluster


def dask(hardware='single', client=None, processes=False, n_workers=1,
         threads_per_worker=1):
    r"""Dask backend initialization.

    Create connection to drive computations using Dask distributed.

    Parameters
    ----------
    hardware : :obj:`str`, optional
        Hardware used to run Dask distributed. Currently available options
        are ``single`` for single-machine distribution.
    client : :obj:`str`, optional
        Name of scheduler (use ``None`` for ``hardware=single``)n.
    processes : :obj:`str`, optional
        Whether to use processes (``True``) or threads (``False``).
    n_workers : :obj:`int`, optional
        Number of workers
    threads_per_worker : :obj:`int`, optional
        Number of threads per each worker

    Returns
    -------
    client : :obj:`dask.distributed.client.Client`

    Raises
    ------
    ValueError
        If ``hardware`` is not ``single``

    """
    if hardware == 'single':
        cluster = LocalCluster(processes=processes, n_workers=n_workers,
                               threads_per_worker=threads_per_worker)
    else:
        cluster = client
    client = Client(cluster)
    return client