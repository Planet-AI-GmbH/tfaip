import os

this_dir = os.path.dirname(os.path.realpath(__file__))


def get_workdir(name: str, *args):
    scenario_dir = os.path.join('test', 'scenario')
    assert(scenario_dir in name)
    base_dir = name[:name.find('/', name.rfind(scenario_dir) + len(scenario_dir) + 1)]
    wd = os.path.join(base_dir, 'workdir')
    assert(os.path.exists(wd))
    assert(os.path.isdir(wd))
    return os.path.join(wd, *args)
