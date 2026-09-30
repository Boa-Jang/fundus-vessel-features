# EyeLiner (bundled)

Minimal EyeLiner + LightGlue code copied from https://github.com/QTIM-Lab/EyeLiner
for local pairwise fundus registration. Original license (Apache 2.0) applies.

## Contents

```
EyeLiner/
├── eyeliner/            # pairwise registration API (EyeLinerP)
│   ├── __init__.py
│   ├── detectors.py     # SPLG + LoFTR keypoint detectors
│   ├── eyeliner.py      # EyeLinerP class
│   └── utils.py         # TPS, coord normalization
└── lightglue/           # LightGlue matcher + SuperPoint/DISK/SIFT (upstream)
    ├── lightglue.py
    ├── superpoint.py
    ├── sift.py, disk.py, aliked.py
    ├── utils.py
    └── viz2d.py
```

## Usage (from 02-alignment.ipynb)

```python
import sys
from pathlib import Path

# EyeLiner + lightglue 를 sys.path 에 추가
EYELINER_ROOT = Path("../EyeLiner").resolve()
if str(EYELINER_ROOT) not in sys.path:
    sys.path.insert(0, str(EYELINER_ROOT))

# eyeliner/detectors.py 가 `from .lightglue import ...` 로 하지만
# lightglue 는 자매 폴더 → sys.modules 로 우회
import importlib
if "eyeliner.lightglue" not in sys.modules:
    sys.modules["eyeliner.lightglue"]       = importlib.import_module("lightglue")
    sys.modules["eyeliner.lightglue.utils"] = importlib.import_module("lightglue.utils")

from eyeliner import EyeLinerP
api = EyeLinerP(kp_method="splg", reg="affine",
                image_size=(3, 256, 256), device="cuda")
theta, cache = api({"fixed_input": tf, "moving_input": tm})
```

## 원본에서 안 가져온 것

- `scripts/`, `misc/`, `assets/` — 훈련·데모 스크립트 (필요 없음)
- `pyproject.toml`, `poetry.lock` — dependency 는 상위 프로젝트에서 이미 설치
- 훈련 데이터 관련 코드 (`data.py`, `pairwise_registrator.py` 등 CLI)
