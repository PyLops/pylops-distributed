from dask.distributed import Client, LocalCluster


try:
    from dask_jobqueue import PBSCluster
    jobqueue = True
except:
    jobqueue = False


def dask(hardware='single', client=None, processes=False, n_workers=1,
         threads_per_worker=1, **kwargscluster):
    r"""Dask backend initialization.

    Create connection to drive computations using Dask distributed.

    Parameters
    ----------
    hardware : :obj:`str`, optional
        Hardware used to run Dask distributed. Currently available options
        are ``single`` for single-machine distribution, ``ssh`` for
        SSH-bases multi-machine distribution and ``pbs`` for
        PBS-bases multi-machine distribution
    client : :obj:`str`, optional
        Name of scheduler (use ``None`` for ``hardware=single``).
    processes : :obj:`str`, optional
        Whether to use processes (``True``) or threads (``False``).
    n_workers : :obj:`int`, optional
        Number of workers
    threads_per_worker : :obj:`int`, optional
        Number of threads per each worker
    kwargscluster:
        Additional parameters to be passed to the cluster creation routine

    Returns
    -------
    client : :obj:`dask.distributed.client.Client`
        Client
    cluster :
        Cluster

    Raises
    ------
    NotImplementedError
        If ``hardware`` is not ``single``, ``ssh``, or ``pbs``

    """
    if hardware == 'single':
        cluster = LocalCluster(processes=processes, n_workers=n_workers,
                               threads_per_worker=threads_per_worker)
    elif hardware == 'ssh':
        cluster = client
    elif hardware == 'pbs':
        if jobqueue == False:
            raise ModuleNotFoundError('dask-jobqueue not installed. ' \
                                      'Run "pip install dask-jobqueue".')
        cluster = PBSCluster(**kwargscluster)
        cluster.scale(jobs=n_workers)
    else:
        raise NotImplementedError('hardware must be single, ssh, or pbs')
    client = Client(cluster)
    return client, cluster
