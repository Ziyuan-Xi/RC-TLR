# RC-TLR

Official release package for **Risk-Controlled Target-Latent Retrieval
(RC-TLR)** on the audited small-data vehicle exterior reconstruction
benchmark.

RC-TLR combines target-latent retrieval with nested, validation-only risk
control. The controller falls back to flat-L2 design-case retrieval when the
learned retrieval space shows collapse or insufficient validation gain. 


## Installation

Python 3.10 is the reference version.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```


Install the PyTorch build appropriate for your CPU/CUDA platform, then:

```bash
pip install -r requirements.txt
```

## Public smoke check

```bash
python tools/make_synthetic_sample.py
python tools/audit_dataset.py \
  --conditions data/synthetic/conditions \
  --targets data/synthetic/targets \
  --output-json data/synthetic/audit.json \
  --output-csv data/synthetic/manifest.csv
```

To re-audit the complete local research data and calculate checksums:

```bash
python tools/audit_dataset.py \
  --conditions _release_assets/dataset_full/conditions \
  --targets _release_assets/dataset_full/targets \
  --meshes _release_assets/dataset_full/obj_meshes \
  --output-json data/dataset_audit.json \
  --output-csv data/dataset_manifest.csv \
  --hash
```

## Reproduce RC-TLR

The checked-in configuration reproduces the conservative ten-fold evaluation:

```bash
python tools/run_rc_tlr.py
```
# Provenance and attribution notice

The local reproduction package from which this repository was organized
identifies the following authors for the original
`IGD_Vehicle_Exterior_Shape` code/data:

- Yuhao Liu
- Maolin Yang
- Pingyu Jiang
- State Key Laboratory of Mechanical Manufacturing Systems,
  Xi'an Jiaotong University
