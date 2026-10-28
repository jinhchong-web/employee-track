# Staff Tracking in a Crowded Video (RT-DETR + BoT-SORT + YOLOE)

Finds the frames in which one specific staff member is present in a crowded, static-camera video (`sample.mp4`) and reports where the person is, as xy pixel coordinates.

A normal tracker cannot follow one person through this clip on its own: it produced **412 separate track IDs**, because an ID breaks every time someone is hidden. This notebook detects and tracks everyone, scores each track fragment for the staff member's small chest tag, and stitches the matching fragments into one chain.

## Tasks

1. Identify which frames in the clip have the staff present.
2. (Bonus) Locate the staff xy coordinates when present.

## Result

Video: 960x720, 25 fps, 1341 frames (53.6 s). After a manual check of the output video (`output/target_result0.mp4`), the correct staff tracklets are **258, 325, 529, 678 and 714**.

| Tracklet | Frames | Time | Detected frames |
|---|---|---|---|
| 258 | 361-393 | 14.44-15.72 s | 33 of 33 |
| 325 | 444-457 | 17.76-18.28 s | 14 of 14 |
| 529 | 827-864 | 33.08-34.56 s | 38 of 38 |
| 678 | 1053-1067 | 42.12-42.68 s | 13 of 15 |
| 714 | 1152-1204 | 46.08-48.16 s | 12 of 53 |

The staff is detected in **110 of 1341 frames (8.2 %, about 4.4 s)**. Frames outside these ranges mean "not detected"; the person may still be in the scene but hidden or missed by the detector. Frame numbers are 0-based.

The automatic stitcher links 10 tracklets in total. Tracklets 39, 355, 362, 812 and 841 were not confirmed as the staff member and are excluded (see [Using the confirmed tracklets](#using-the-confirmed-tracklets)).

### Coordinates (first and last detected box of each tracklet)

All values are pixels in the 960x720 frame: (0, 0) is the top-left corner, x grows to the right (0-960), y grows downward (0-720).

| Tracklet | Frame | Event | center (x, y) | foot (x, y) |
|---|---|---|---|---|
| 258 | 361 | start | (577.3, 302.7) | (577.3, 370.9) |
| 258 | 393 | end | (567.7, 326.4) | (567.7, 377.5) |
| 325 | 444 | start | (567.5, 255.4) | (567.5, 330.5) |
| 325 | 450 | seed A | (530.7, 144.4) | (530.7, 245.8) |
| 325 | 457 | end | (444.9, 80.4) | (444.9, 160.7) |
| 529 | 827 | start | (560.3, 250.3) | (560.3, 334.2) |
| 529 | 850 | seed B | (646.6, 105.9) | (646.6, 177.6) |
| 529 | 864 | end | (545.7, 257.2) | (545.7, 327.5) |
| 678 | 1053 | start | (529.5, 663.6) | (529.5, 720.0) |
| 678 | 1067 | end | (529.3, 539.3) | (529.3, 620.6) |
| 714 | 1152 | start | (514.4, 658.5) | (514.4, 720.0) |
| 714 | 1204 | end | (641.1, 83.8) | (641.1, 134.9) |

- `center` is the middle of the person's bounding box: `((x1 + x2) / 2, (y1 + y2) / 2)`.
- `foot` is the middle of the box's bottom edge: `((x1 + x2) / 2, y2)`, roughly where the person stands. A foot y of 720 means the box touches the bottom edge of the frame.
- The notebook prints xy only at lock and lost events. Per-frame boxes are stored in `tracklet_boxes[id][frame] = (x1, y1, x2, y2)`; the same formulas give an xy for every detected frame.

## How it works

```
sample.mp4
  -> RT-DETR-X          detect every person
  -> BoT-SORT           link detections into short tracklets
  -> YOLOE (visual      score each tracklet for the staff chest tag
     prompt)
  -> stitching          chain tracklets from two user-given seeds
  -> lock report + annotated video (output/target_result0.mp4)
```

The staff member changes outfit during the clip (white top at 18 s, black jacket at 34 s) but the chest tag stays, so the tag is used as the identity clue. One YOLOE tag model is built per outfit (A and B).

### Models

| Model | Weights | Role |
|---|---|---|
| RT-DETR-X | `rtdetr-x.pt` | Transformer person detector (class 0 only). Attention over the whole image copes well with crowds and partial occlusion, and no NMS step is needed. |
| BoT-SORT | `botsort_custom.yaml` (written by Cell 2) | Assigns IDs across frames using motion and appearance (ReID). |
| YOLOE-11l-seg | `yoloe-11l-seg.pt` | Open-vocabulary model prompted with an example image of the tag instead of a text label. |

### Key parameters (Cell 2)

| Parameter | Value | Why |
|---|---|---|
| `CONF` | 0.15 | Low on purpose, so partly hidden people are still detected |
| `IMGSZ` | 960 | Equals the video width, so distant people keep detail |
| `track_buffer` | 200 | A lost track is kept for 8 s before deletion |
| `gmc_method` / `with_reid` | none / True | Static camera, so no motion correction; appearance matching reduces ID swaps |
| `MIN_TRACKLET_LEN`, `N_SAMPLES` | 8, 15 | Ignore tiny fragments; score 15 evenly spaced frames per tracklet |
| `TORSO_SCALE` | 3 | The chest crop (12-55 % of box height) is enlarged because the tag is tiny |
| `MAX_GAP`, `SHORT_GAP` | 300, 25 frames | Link fragments up to 12 s apart; gaps up to 1 s may link by motion alone |
| `V_MAX`, `GATE_BASE_H` | 10 px/frame, 0.5 | Candidate must lie within 0.5 x person height + 10 px x gap frames |
| `TAG_ACCEPT_RATIO` | 0.7 | Long-gap links need a tag score of at least 0.7 x the weaker seed score |
| `ANCHOR_BONUS` | 1.0 | Prefer linking to the other user-confirmed seed |
| `W_TAG`, `W_SPACE`, `W_GAP` | 1.0, 0.5, 0.2 | Link score = tag + 0.5 x closeness - 0.2 x gap/300 |

### Tag functions (Cell 9)

- `make_tag_model(refer_img, box)` builds a YOLOE model whose only class is "the tag". It receives a reference torso crop and the box drawn around the tag, passes them as a visual prompt and runs one prediction so YOLOE stores what the tag looks like. Called twice: `yoloe_A` (white top) and `yoloe_B` (black jacket).
- `_best_conf(m, torso_img)` runs one tag model on one torso crop with a very low threshold (0.001) and returns the highest confidence found, or 0.0 if nothing is found. The notebook needs the raw score, not a yes/no.
- `tag_scores_AB(torso_img)` returns the score against tag A and tag B. Cell 10 applies it to up to 15 frames per tracklet and keeps the maximum (`best_A`, `best_B`, `tag_best`).

## Setup

Developed on Windows with PyTorch 2.12.1 (CUDA) and an NVIDIA RTX 5060 Laptop GPU. A CUDA GPU is strongly recommended; the code falls back to CPU, which will be slow.

```bash
# install PyTorch first (pick the build for your CUDA version: https://pytorch.org)
pip install -U ultralytics pillow opencv-python pandas matplotlib tqdm jupyter
```

Put the input video next to the notebook as `sample.mp4` (or change `VIDEO_PATH` in Cell 2). Ultralytics downloads the model weights (`rtdetr-x.pt`, `yoloe-11l-seg.pt`) on first use. If VRAM is tight, `rtdetr-l.pt` and `yoloe-11s-seg.pt` are smaller alternatives named in the config cell.

## Running the notebook

Cells are counted from 1, in run order.

1. **Cells 1-6**: imports, config, video info, tracking (this creates the tracklets) and a preview of the IDs at the two seed moments (18 s and 34 s, set by `SEED_A_SECOND` and `SEED_B_SECOND`).
2. **Cell 7** asks for the staff member's tracklet ID at each seed moment. In this run: `325` (white top) and `529` (black jacket).
3. **Cell 8** asks for the tag box inside each torso crop as `x1,y1,x2,y2` in crop pixels. In this run: `100,100,150,150` and `200,75,250,125`.
4. **Cells 9-10**: build the two tag models and rank every tracklet by tag score.
5. **Cell 11**: stitch the chain of tracklets from the two seeds.
6. **Cell 12**: print the lock report (LOCK / LOST / RE-LOCK with `center` and `foot`).
7. **Cell 13**: write the annotated video `output/target_result0.mp4` (red box and `TARGET | trk <id>` label when detected, `SEARCHING...` otherwise).

### Using the confirmed tracklets

The tracklet IDs 258, 325, 529, 678 and 714 were verified manually by watching the output video (`output/target_result0.mp4`). The stitcher's chain also contains five tracklets that were not confirmed in that check (39, 355, 362, 812, 841), which is why Cell 13 reports 289 detected frames instead of 110. To keep only the verified ones, set the chain before running Cells 12 and 13:

```python
chain_ids = [258, 325, 529, 678, 714]
```

## Limitations

- The tag scores are low (about 0.01-0.08), so they work as relative evidence, not a clear yes/no. In the ranking (Cell 10) the white-top seed ranks 12th, so the tag alone cannot find the staff member.
- Because the tag term is small, choosing between candidate tracklets depends mostly on position and time gap, not on the tag.
- The method needs two manual seeds and two manual tag boxes; the stitched chain must be checked against the video.
- Only 8.2 % of frames have a detected box. The tracker can lose the person when they are hidden, and frames without a box are not proof that the staff is absent.

## Files

| File | Description |
|---|---|
| `transformer_v3.ipynb` | The full pipeline (detection, tracking, tag scoring, stitching, video output) |
| `sample.mp4` | Input video (add your own copy) |
| `output/target_result0.mp4` | Annotated result video written by Cell 13 |
