from dataclasses import dataclass


@dataclass
class Resource:
    id: str
    rel_path: str
    basename: str = None
    dump_dir: str = ''
    dump_path: str = None
    abs_path: str = None
