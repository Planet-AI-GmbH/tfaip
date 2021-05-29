# Copyright 2021 The tfaip authors. All Rights Reserved.
#
# This file is part of tfaip.
#
# tfaip is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# tfaip is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# tfaip. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
"""A pool of processes derived from multiprocessing.Pool with a custom worker function"""
import multiprocessing
import logging
import sys


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

            if "tensorflow" in sys.modules:
                logger.warning(
                    "You imported tensorflow at some point in the parallel pipeline. "
                    "Running the code will work, but consume more memory and takes more time for "
                    "initialization. Consider to remove all tensorflow imports from your data workers. "
                    "(Watch out for import in __init__.py cause they are done automatically."
                    "For Debugging I recommend to set a breakpoint in tensorflow.__init__.py. "
                    "The first stop is legitimate (import from main thread), any other import is from a "
                    "spawned child. Try to track down the import history and find the bad boy that causes "
                    "to import tensorflow."
                    "Note: Use local imports if you must import tensorflow."
                )
        return Initializer.worker


class WrappedPool(multiprocessing.pool.Pool):
    def __init__(self, worker_constructor, **kwargs):
        super().__init__(initializer=Initializer(worker_constructor), **kwargs)

    def imap_gen(self, gen):
        return self.imap(worker_func, gen)


def worker_func(*args, **kwargs):
    # Calls process on the actual worker that was created by the initializer as global var for the python process
    return Initializer.worker.process(*args, **kwargs)
