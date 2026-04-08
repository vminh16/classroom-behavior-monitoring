# Runtime Artifacts

This public repository does not track large binary assets such as:

- Paddle inference models
- sample videos
- generated benchmark outputs
- temporary review files

## Download Link

Replace the placeholder below with your shared Google Drive folder before publishing:

- picodet-m-416: https://drive.google.com/drive/folders/1fUPfqDetWC0KeGNB-3Vz8tQsHdNqgrQi?usp=drive_link
- pplcnet: https://drive.google.com/drive/folders/1Xpu1lKiybhJUunUpFQULtiHCtB25SxPg?usp=drive_link
- ppTSM-fight: https://drive.google.com/drive/folders/1qSPHCpGJ25jXzW1DLbRUjz3n5h5ys05o?usp=drive_link
## Expected Layout

After downloading the artifact bundle, extract it so the project contains these paths:

```text
output_inference/
  picodet_m_416_classroom/
    infer_cfg.yml
    model.pdmodel
    model.pdiparams
    model.pdiparams.info
  pplcnet_behavior/
    infer_cfg.yml
    inference.pdmodel
    inference.pdiparams
    inference.pdiparams.info
  ppTSM_fight/
    ppTSM/
      model.pdmodel
      model.pdiparams
      model.pdiparams.info

test_data/
  data.mp4
  classroom_test.mp4
  test.mp4
```

## Notes

- `output_inference/` is the canonical runtime artifact location for this repository.
- `deploy_bundle/` reuses the same artifacts from `output_inference/` to avoid storing duplicate model copies.
- `models/` and `deploy_bundle/models/` are kept only as legacy local mirrors on the author's machine and are ignored by default.
- `output/` contains generated benchmark videos, logs, and figures and should not be committed.
- `tmp/` is scratch space and should not be committed.

## Minimal Setup

If you only want to run the deploy bundle, you still need:

- `output_inference/picodet_m_416_classroom`
- `output_inference/pplcnet_behavior`
- optionally `test_data/data.mp4` if you want the default demo input
