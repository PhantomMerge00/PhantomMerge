# Full replication

```bash
export PHANTOM_MERGE_ROOT="$(pwd)"
source reproduce/env.sh
pip install -r reproduce/requirements-ccer.txt
pip install -e third_party/jacobian-lens
bash reproduce/run_paper_tables.sh
```

Shopping end-to-end: `reproduce/run_shopping_pipeline.sh`. τ³ domain: `reproduce/run_tau3_pipeline.sh telecom`.

Large files: `reproduce/DATA_SYNC_MANIFEST.json`.
