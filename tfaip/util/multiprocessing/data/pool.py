# Copyright 2020 The tfaip authors. All Rights Reserved.
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
