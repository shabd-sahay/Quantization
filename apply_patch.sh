#!/usr/bin/env bash
# Run from the project root. Makes src/mlp.py import Scaler from the new numpy-only module,
# so the Streamlit app never pulls in torch.
set -e
python - << 'PY'
import re
from pathlib import Path
p = Path("src/mlp.py"); s = p.read_text()
if "from src.scaler import Scaler" in s:
    print("already patched"); raise SystemExit
# strip the Scaler class (from 'class Scaler' up to 'class MLP')
new = re.sub(r"class Scaler:.*?(?=class MLP)", "", s, flags=re.S)
if new == s:
    raise SystemExit("could not find the Scaler class — patch manually")
new = new.replace("import torch.nn as nn",
                  "import torch.nn as nn\n\nfrom src.scaler import Scaler  # re-exported: existing imports keep working")
p.write_text(new)
import ast; ast.parse(new)
print("patched src/mlp.py")
PY
