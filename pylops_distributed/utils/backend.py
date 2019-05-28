try:
    from dask.distributed import Client, LocalCluster
except ModuleNotFoundError:
    Client = None


def dask(hardware='single', processes=True, threads_per_worker=None):
    r"""Dask backend initialization.

    Create connection to drive computations using Dask distributed.

    Parameters
    ----------
    hardware : :obj:`str`, optional
        Hardware used to run Dask distributed. Currently available options
        are ``single`` for single-machine distribution.
    processes : :obj:`str`, optional
        Whether to use processes (``True``) or threads (``False``).
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
        cluster = LocalCluster(processes=processes,
                            threads_per_worker=threads_per_worker)
        client = Client(cluster)
    return client