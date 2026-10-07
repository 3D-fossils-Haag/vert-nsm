cd ~/NSM/nsm/paper && mkdir -p splits && python -c "
import json, re
norm = lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower())
import pandas as pd
specs = sorted(pd.read_csv('../lizard_species_list.csv').specimen.dropna(), key=len, reverse=True)
c = json.load(open('../run_v72/model_params_config.json'))
keys = ['list_mesh_paths', 'val_paths', 'test_paths']
d = {k: sorted(c[k]) for k in keys}
import os
match = lambda m: next((s for s in specs if norm(os.path.basename(m)).startswith(norm(s))), None)
d['specimens'] = sorted({s for k in keys for m in d[k] if (s := match(m))})
json.dump(d, open('splits/downsample_95spec.json', 'w'), indent=1)
print(len(d['specimens']), 'specimens,', sum(len(d[k]) for k in keys), 'meshes')"