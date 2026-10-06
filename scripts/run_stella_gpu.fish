#!/usr/bin/env fish

cd ~/PycharmProjects/Stella
source .venv/bin/activate.fish

set cuda_libs (python -c 'import importlib.util; mods=["nvidia.cublas.lib","nvidia.cudnn.lib"]; paths=[]
for m in mods:
    spec=importlib.util.find_spec(m)
    if spec is None:
        continue
    if spec.submodule_search_locations:
        paths.append(str(list(spec.submodule_search_locations)[0]))
    elif spec.origin:
        import os
        paths.append(os.path.dirname(spec.origin))
print("\n".join(paths))')

set -gx LD_LIBRARY_PATH $cuda_libs $LD_LIBRARY_PATH

python -m stella.ui_app
