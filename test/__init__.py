import os

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable gpu usage

this_dir = os.path.dirname(os.path.realpath(__file__))
workdir_dir = os.path.join(this_dir, 'workdirs')


def get_workdir(*args):
    return os.path.join(workdir_dir, *args)
