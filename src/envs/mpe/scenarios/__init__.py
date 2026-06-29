import importlib.machinery
import importlib.util
import os.path as osp


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    loader = importlib.machinery.SourceFileLoader("scenario_module", pathname)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module