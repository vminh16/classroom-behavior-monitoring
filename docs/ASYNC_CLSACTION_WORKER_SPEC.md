# Async Classifier Worker Specification

## Status

- Drafted before implementation.
- Target branch: `codex/async-cls-worker`
- Backup snapshot commit: `48a1c89`

## Problem Statement

The current live pipeline executes `MOT -> crop -> cls_action -> behavior_filter -> behavior_state_machine -> render/publish` on the same per-frame critical path in [deploy/pipeline/pipeline_product.py](/C:/Users/USER/classroom-behavior-monitoring/deploy/pipeline/pipeline_product.py:1799).

This keeps behavior semantics simple, but it also means the frame loop waits for `cls_action` every time [deploy/pipeline/pipeline_product.py](/C:/Users/USER/classroom-behavior-monitoring/deploy/pipeline/pipeline_product.py:1895). Local measurements on this repo showed:

- `MOT-only` is materially faster than full behavior pipeline.
- `cls_action` is the main incremental cost.
- existing live capture already uses a latest-frame queue of size 1, so the main pain point is critical-path cadence, not unbounded app-side frame buffering.

## Goals

- Remove `cls_action` from the main frame-loop critical path.
- Preserve the current per-track output contract for visualization:
  - `result['cls_action']` remains a `dict[int, dict]` or equivalent id-to-result mapping.
- Preserve existing behavior filtering and alert semantics as closely as possible.
- Make the feature opt-in and rollback-safe through config.
- Keep changes local to the deploy runtime used by the repo:
  - [deploy/pipeline/pipeline_product.py](/C:/Users/USER/classroom-behavior-monitoring/deploy/pipeline/pipeline_product.py:1)
  - supporting modules under [deploy/pipeline](/C:/Users/USER/classroom-behavior-monitoring/deploy/pipeline)

## Non-Goals

- No attempt to implement Paddle C++ external stream control from Python.
- No rewrite of detector or tracker.
- No model retraining.
- No change to Web UI feature scope beyond compatibility with the runtime.

## Two Candidate Designs

### Option A: Whole-frame FIFO async worker

Pipeline:

`main thread: MOT + crop -> enqueue frame task`

`worker: cls_action + filtering + state machine`

`main thread: render using latest completed packet`

Complexity:

- Main critical path: `O(F * (D + T + R))`
- Worker path: `O(F * T * C_cls)`
- Memory: `O(Q + T)`

Pros:

- Simple mental model.
- Minimal state stitching in main thread.

Cons:

- FIFO queues preserve stale work and can rebuild latency.
- A slow worker can keep producing old results that are already visually obsolete.

### Option B: latest-frame/latest-result async worker with bounded queue

Pipeline:

`main thread: MOT + crop -> replace pending task -> render from last known per-track cache`

`worker: cls_action + filtering + state machine -> publish result packet`

Complexity:

- Main critical path: `O(F * (D + T + R))`
- Worker path: `O(F * T * C_cls)` in the worst case
- Memory: `O(1 + T + B)`
  - `1` pending task
  - `T` cached track results
  - `B` small buffered frames for delayed alert snapshots

Pros:

- Bounded latency behavior.
- Easier to keep preview fresh.
- Matches the repo’s existing live-source philosophy (`live_queue_size=1`).

Cons:

- Harder state management.
- Requires stale-result rejection and buffered notification frames.

## Chosen Design

Option B.

Reason:

- The repo already prefers freshness over completeness for live sources.
- For operator-facing realtime preview, stale classifier work is less valuable than current MOT state.
- A size-1 pending queue gives a predictable upper bound on async backlog.

## Key Constraints From Current Repo

1. `cls_action` output format must remain compatible with:
   - [deploy/pipeline/minimal_visualize.py](/C:/Users/USER/classroom-behavior-monitoring/deploy/pipeline/minimal_visualize.py:27)
   - [deploy/pipeline/pipeline_product.py](/C:/Users/USER/classroom-behavior-monitoring/deploy/pipeline/pipeline_product.py:2153)

2. `ActionVisualHelper.update()` expects iterable `(mot_id, action_res)` pairs:
   - [deploy/pipeline/pphuman/action_utils.py](/C:/Users/USER/classroom-behavior-monitoring/deploy/pipeline/pphuman/action_utils.py:108)

3. `BehaviorFilter` and `BehaviorStateMachine` are stateful and keyed by track id:
   - [deploy/pipeline/behavior_filter.py](/C:/Users/USER/classroom-behavior-monitoring/deploy/pipeline/behavior_filter.py:11)
   - [deploy/pipeline/behavior_state_machine.py](/C:/Users/USER/classroom-behavior-monitoring/deploy/pipeline/behavior_state_machine.py:39)

4. Existing sync implementation mutates predictor caches directly during reset/cleanup:
   - [deploy/pipeline/pipeline_product.py](/C:/Users/USER/classroom-behavior-monitoring/deploy/pipeline/pipeline_product.py:882)
   - [deploy/pipeline/pipeline_product.py](/C:/Users/USER/classroom-behavior-monitoring/deploy/pipeline/pipeline_product.py:892)

5. Telegram notifications currently use the frame being processed at the time events are generated:
   - [deploy/pipeline/pipeline_product.py](/C:/Users/USER/classroom-behavior-monitoring/deploy/pipeline/pipeline_product.py:1148)

## Architecture

### New module

Create a new module under [deploy/pipeline](/C:/Users/USER/classroom-behavior-monitoring/deploy/pipeline):

- `async_cls_action.py`

### New runtime pieces

1. `ClsActionPostProcessor`

Responsibilities:

- normalize raw classifier output to `dict[int, dict]`
- apply temporal voting
- apply state machine
- cleanup stale tracks
- expose reset/drop-track helpers

This becomes the single source of truth for postprocessing behavior labels in both sync and async modes.

2. `AsyncClsActionWorker`

Responsibilities:

- own a dedicated `ClsActionRecognizer`
- own its own `ClsActionPostProcessor`
- run on a background thread
- consume bounded latest-task queue
- produce completed packets into a result queue

3. Pipeline-side caches

- `self._async_cls_cache`: latest per-track classifier result
- `self._async_cls_frame_buffer`: small ring buffer of RGB frames keyed by `frame_id` for delayed notifications
- `self.cls_action_async_worker`: worker instance or `None`

## Data Contracts

### Task

- `frame_id: int`
- `now_ts: float`
- `mot_res: dict`
- `crop_input: list[np.ndarray]`
- `mode: "infer" | "cleanup"`

### Output packet

- `frame_id: int`
- `now_ts: float`
- `results: dict[int, dict]`
- `events: list[dict]`
- `stale_track_ids: list[int]`

## Config Contract

Add an optional nested block under `ID_BASED_CLSACTION`:

```yaml
async_worker:
  enable: false
  queue_size: 1
  frame_buffer_size: 32
  worker_device: same
  worker_run_mode: same
```

Rules:

- `enable=false`: old synchronous path stays unchanged.
- `queue_size` will be clamped to `1..4`, but implementation will optimize for `1`.
- `worker_device`:
  - `same`: inherit pipeline device
  - `cpu`: force classifier worker onto CPU
  - `gpu`: force GPU
- `worker_run_mode`:
  - `same`: inherit main `--run_mode`
  - otherwise use explicit override such as `paddle`, `trt_fp32`, `trt_fp16`

## Important Operational Risk

If both main pipeline and worker run on GPU in the same Python process, true overlap is not guaranteed.

Reason:

- Paddle’s documented GPU multi-stream control is described for C++ API, and the default strategy uses a process-level stream shared by predictors.
- Therefore, `async` in Python can reduce host-side blocking and improve preview freshness, but it may still serialize GPU work if both predictors target the same default stream.

Implication:

- `worker_device=cpu` is expected to be the safest decoupling option when main `MOT` runs on GPU.
- `worker_device=same` remains available for experimentation and compatibility.

## Behavioral Semantics

### Sync mode

- unchanged public behavior
- internal implementation may route through `ClsActionPostProcessor`

### Async mode

- visualization uses latest known per-track classifier result
- a new track may temporarily show no classifier label until the first worker result lands
- alerts are emitted using the buffered frame closest to the worker result’s `frame_id`
- stale worker results older than the latest consumed packet are ignored

## Integration Points

### Pipeline init

When `ID_BASED_CLSACTION.enable` is true:

- build labels as today
- create `ActionVisualHelper` as today
- branch on `async_worker.enable`

If async is enabled:

- build `AsyncClsActionWorker`
- start worker before video loop
- do not use sync `apply_behavior_filter()` on the critical path

If async is disabled:

- keep old direct call path

### Video loop

For each frame:

1. drain completed async packets
2. store current frame in bounded frame buffer
3. if there are MOT crops:
   - submit async inference task
4. else:
   - submit cleanup task
5. build current `cls_action_res` from `self._async_cls_cache` filtered by visible track ids
6. update `pipeline_res`
7. visualize using current cache

### Reset/reconnect/finally

- reset worker-owned state via worker API, not by mutating worker predictor internals from the main thread
- stop worker cleanly with `Event` + `join()`

## Edge Cases To Preserve

1. zero visible tracks
2. stream reconnects
3. track id disappears before classifier result returns
4. worker queue full while source remains live
5. Telegram sink enabled
6. visualization style `minimal`
7. `cls_action_res` consumers expecting `dict` semantics

## Testing Plan

1. Static validation

- `python -m compileall` on touched runtime files

2. Functional smoke tests

- sync path still runs with `async_worker.enable=False`
- async path runs on `video_file`
- async path runs on `rtsp` dry-run or simulated source

3. Correctness checks

- no exception on zero-track frames
- delayed worker results do not crash visualization
- stale tracks are removed from async cache
- reconnect resets async state

4. Performance checks

- compare sync vs async on the same sample
- record:
  - processing FPS
  - render FPS
  - frame age
- success criterion:
  - no regression in sync mode
  - async mode keeps preview/result updates functional and does not increase unbounded lag

## Rollback Plan

- set `ID_BASED_CLSACTION.async_worker.enable=false`
- no migration needed for artifacts or model files

## Implementation Notes

- Prefer a new module over large in-place threading logic inside `pipeline_product.py`.
- Prefer a reusable postprocessor over duplicating `apply_behavior_filter()` logic.
- Keep worker thread non-daemonic and stop it with `threading.Event`.
- Use a bounded queue and latest-task replacement, not an unbounded FIFO.
