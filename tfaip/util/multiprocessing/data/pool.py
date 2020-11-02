import multiprocessing
import logging


logger = logging.getLogger(__name__)


class Initializer:
    """
    Global variable worker per thread that is used to call the actual method
    """
    worker = None

    def __init__(self, worker_creator):
        self.worker_creator = worker_creator

    def __call__(self, *args, **kwargs):
        if not Initializer.worker:
            logger.debug("Initializing Worker")
            Initializer.worker = self.worker_creator()
            Initializer.worker.initialize_thread()
        return Initializer.worker


class WrappedPool(multiprocessing.pool.Pool):
    def __init__(self, worker_constructor, **kwargs):
        super(WrappedPool, self).__init__(initializer=Initializer(worker_constructor), **kwargs)

    def imap_gen(self, gen):
        return self.imap(worker_func, gen)


def worker_func(*args, **kwargs):
    # Calls process on the actual worker that was created by the initializer as global var for the python process
    return Initializer.worker.process(*args, **kwargs)
