import sys
import inspect
import collections

# 1. Fix inspect.getargspec for Python 3.11+
if not hasattr(inspect, 'getargspec'):
    ArgSpec = collections.namedtuple('ArgSpec', ['args', 'varargs', 'keywords', 'defaults'])
    def getargspec(func):
        full = inspect.getfullargspec(func)
        return ArgSpec(full.args, full.varargs, full.varkw, full.defaults)
    inspect.getargspec = getargspec

# 2. Fix NumPy deprecated aliases before chumpy imports them
import numpy as np
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'object'):
    np.object = object

# 3. Add to sys.modules so 'from numpy import int' works seamlessly
sys.modules['numpy'].int = int
sys.modules['numpy'].float = float
sys.modules['numpy'].bool = bool
sys.modules['numpy'].object = object


import pickle
import numpy as np

import inspect

# Monkey-patch the missing function for Python 3.12 compatibility
if not hasattr(inspect, 'getargspec'):
    import collections

    ArgSpec = collections.namedtuple('ArgSpec', ['args', 'varargs', 'keywords', 'defaults'])


    def getargspec(func):
        full = inspect.getfullargspec(func)
        return ArgSpec(full.args, full.varargs, full.varkw, full.defaults)


    inspect.getargspec = getargspec

# Now your other imports can safely run
import chumpy


# Path to your official legacy SMPL pkl file
old_pkl_path = "models/smpl/SMPL_NEUTRAL.pkl"
new_pkl_path = "models/smpl_neutral_python3.pkl"

# 1. Open with 'latin1' encoding to prevent UnicodeDecodeErrors in Python 3
with open(old_pkl_path, 'rb') as f:
    src_data = pickle.load(f, encoding='latin1')

output_data = {}

# 2. Extract data and force chumpy objects back into standard NumPy arrays
for k, v in src_data.items():
    if type(v).__name__ == 'chumpy' or hasattr(v, 'v'):
        output_data[k] = np.array(v.v)  # Extract the raw array from chumpy
    elif isinstance(v, dict):
        # Handle sub-dictionaries if present
        output_data[k] = {nk: (np.array(nv.v) if hasattr(nv, 'v') else nv) for nk, nv in v.items()}
    else:
        output_data[k] = v

# 3. Save a clean, Python 3-native version
with open(new_pkl_path, 'wb') as f:
    pickle.dump(output_data, f, protocol=2)

print(f"Successfully created a chumpy-free model at: {new_pkl_path}")
