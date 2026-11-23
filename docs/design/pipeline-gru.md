---
status: draft
last-reviewed: 2026-11-23
---
# Pipeline: landmarks to GRU

## Scope
Selected landmarks, normalization, feature vector, window length/FPS, model, ONNX export.

## Current implementation (MLP)
- 20 landmarks (see `src/core/config.py`), centroid-normalized, 60-dim vector, `src/models/wakeguard_mlp.onnx`.
- Out of date as a description of the model: `main.py` still runs this MLP, but the model is now GRU v3
  ("Final model" below, `src/models/eye_gru_v3`); wiring it into `main.py` is pending.

## Thesis target

### Decided (2026-11-17)
- **Target**: per-frame probability that the driver is **in a drowsy episode**, from eyes and head only.
  Not "eye closed now": NTHU-DDD labels the drowsy state (slow blinks, droop, nod, closures), and labels made
  from an `eyeBlink` threshold would only teach the model that threshold.
- **Outside the model, as rules running beside it**: yawn (`jawOpen` held), head drop with face lost, eyes closed
  ≥ ~1.5 s → critical (fast path). (Out of date: the proposed alert rule of 2026-11-23 uses closure ≥ 2.0 s,
  see `scoring-water-tank.md`.)
- **Labels**
  - Own sessions (`data/raw/VPMT`): `eyes_half`, `eyes_closed`, `nodding` → drowsy; `normal`, `talking` → awake;
    `yawn` and `unlabeled` excluded. Actions lag the script by 1–3 s (checked on video), so evaluation skips the
    first 1.5 s of every step.
  - NTHU-DDD (Kaggle mirror, `data/processed/nthu`): `drowsy` / `notdrowsy` per frame as given. `yawning`
    sequences are kept out of the eye model and used to tune the yawn rule (their labels are aligned in time:
    `jawOpen` p95 0.62 drowsy vs 0.01 not).
  - UTA-RLDD (`data/processed/uta`): one self-reported label per ~10 min video; used only for a long-window
    fatigue level, never as per-frame targets.
- **Folds**: the unit is the person (S07's two sessions stay together). Eye model: leave-one-person-out over the
  13 people of NTHU (4) + own (9), hyper-parameters chosen by an inner leave-one-person-out on the 12 others.
  UTA keeps its 5 official subject folds. Results are reported per source. A final test set is recorded later
  with the camera mounted as in the product and is used once, for the report only.
- **Per-person baseline**: from that person's awake segment only (first ~30 s), as the runtime calibration does;
  never from a test person's drowsy frames.
- **Metrics**: frame AUC with transitions ignored; event recall (closures, nods); false alarms per hour on awake
  segments; latency from closure onset to the probability crossing the threshold.

### Order of work (decided 2026-11-19)
Rule baseline first, GRU second, same loader and same leave-one-person-out evaluation for both.
1. Shared loader: session CSV → the 15 per-frame features below, per-person baseline from the awake segment.
2. Labels: own sessions skip the first 1.5 s of each step and ±0.75 s around label changes.
3. B0 rule: `eyeBlink` above the person's baseline + closure duration; tuned on the inner folds only.
4. GRU, compared with B0–B3. If it does not beat B0, the product ships the rule and the GRU is reported as an
   experiment. Why: 13 people only; on UTA fold 1 the signal sat in blink rate, PERCLOS and closure duration,
   all rule-computable; on NTHU only 16% of drowsy frames have closed eyes, the part a rule may miss.

### B0 result (2026-11-19, `python -m tools.eval_b0`, report `data/processed/reports/b0.json`)
- Pooled frame AUC 0.72 over 13 people; NTHU 0.77 (4 people), own 0.59 (9 people), threshold at 5% awake FPR.
- Own sessions are the weak part, and the cause is the labels, not the rule: in the scored part of `eyes_closed`
  steps the median `eyeBlink` rise is about 0 for S06, S08, S10, S12, S13 (a 2.5 s step minus the 1.5 s + 0.75 s
  skips leaves 0.25 s, and the action lags 1-3 s). Own frame labels need realigning before they can rank models.

### Input check (2026-11-19, `python -m tools.eval_inputs`, report `data/processed/reports/inputs.json`)
- Eyelid angle (ELA, arXiv 2511.19519) and eyeBlink velocity were tried as extra inputs. Neither helped: NTHU
  leave-one-person-out with a logistic stand-in model gave pooled AUC 0.725 (15 inputs), 0.726 (+ELA),
  0.724 (+velocity), 0.725 (+both). ELA was not less pitch-sensitive than EAR on awake frames (|r| 0.26 vs 0.25 own,
  0.28 vs 0.25 NTHU); our ELA skips the paper's extra z rescale. Per-frame separability on NTHU: EAR 0.72,
  ELA 0.69, eyeBlink 0.68. Decision: keep the 15 inputs.

### Training setup (decided 2026-11-20, `python -m tools.train_gru`)
- Data on Kaggle (private dataset, packed by `tools/pack_kaggle.py`): NTHU (4
  people) + UTA folds 1-4 (48 people, 143 videos; the Kaggle mirror has no fold 5 and one fold-3 video is missing).
  Own sessions are not uploaded until the subjects' consent covers it.
- Two heads on one GRU (`src/training/gru_model.py`): episode head trained on NTHU frame labels (10 s crops),
  fatigue head trained on UTA video labels as the mean logit over 60 s crops (UTA labels never become frame
  targets). UTA `low_vigilant` is not trained on.
- Folds: outer fold k tests NTHU person k + UTA official fold k; an inner 3-fold split of the training people picks
  epochs, temperatures and the episode threshold (5% FPR on out-of-sample inner predictions). 4 x (3 + 1) trainings.
- UTA baseline per video (first 30 s): videos were recorded separately with the phone placed anew.
- UTA frame size: the extractor did not record it and videos mix portrait and landscape, even within one person
  (UTA04 alert 848x480, drowsy 480x848), which breaks EAR. `tools/uta_aspect.py` writes it into each meta: from
  the video for fold 1, otherwise estimated from rigid 3D landmark ratios (50/50 correct on videos of known size;
  ~10 estimates fall between two candidates, EAR scale error up to ~9% there).

### First GRU run (2026-11-20, Kaggle T4, 13 min)
- NTHU episode head, per-person AUC mean 0.863 (B0 0.753): NTHU001 0.96, NTHU002 0.90, NTHU005 0.78, NTHU006 0.82.
- The threshold does not transfer between people: NTHU001 and NTHU006 never cross it (event recall 0),
  NTHU005 is over it 66% of awake frames (348 false alerts/h). Mean event recall 0.37 vs B0 0.60.
- UTA fatigue head, 60 s window AUC mean 0.70 over the 4 folds (0.59 / 0.70 / 0.71 / 0.80), 35 of 47 people
  with drowsy windows above alert. On fold 1 this is below the hand-made 60 s features (0.78), but those used
  the alert video as baseline for all three videos, while the GRU uses a per-video baseline; not comparable yet.
- Inner-fold epoch choice is unstable (2, 5, 18, 23 epochs); validation curves are flat and noisy.
- Next: a per-driver score offset from the calibration period (evaluated on the saved checkpoints), then the
  water tank on top before reading false alerts/h.

### Second run + bench (2026-11-20, `tools/bench_gru.py`)
- v1 overfit: training loss fell to ~0.05 while validation was flat or falling after epoch ~5 (each "epoch" =
  100 steps ≈ 38 passes over the 3 training NTHU people). v2: 12 epochs, dropout 0.3, weight decay 0.05;
  GRU 48 (12,418 params) and GRU 24 (4,738).
- GRU 48 v2: NTHU AUC mean 0.895 (0.96 / 0.97 / 0.77 / 0.87), UTA fatigue window AUC 0.71. GRU 24: 0.844, 0.72.
- Per-driver offset (episode logit minus its median over the calibration frames) fixes most of the threshold
  transfer problem. Through the current water tank (+5/-2, warning 50), mean over the 4 NTHU people:
  GRU 48 + offset: event recall 0.57, detect 4.3 s, 19 false alerts/h NTHU awake, 6.8/h UTA alert videos;
  B0: 0.54, 16.6 s, 14.5/h, 6.2/h; GRU 48 without offset: 0.28. Same recall as B0, ~4x faster detection.
- Only 24 NTHU events and ~5-10 awake minutes per person: one false alert moves a person's rate by 6-12/h, so
  these differences are indicative, not significant. Most false alerts come from NTHU005 (all methods).

### Pure GRU vs 3 layers (2026-11-20, `tools/compare_arch.py`, `runs/gru_v2/compare_h48.json`)
GRU 48 v2 + per-driver offset, water tank as in `src/core/config.py`; events pooled over the 4 NTHU people
(24 events, only 11.1 awake minutes in total: 1 false alert = 5.4/h); UTA alert videos 7.9 h.

| Arm | NTHU recall | detect | NTHU FA/h any / red | UTA alert FA/h any / red | UTA drowsy videos alerted |
|---|---|---|---|---|---|
| pure (tank: yellow 50, red 100) | 0.67 | 2.3 s | 21.6 / 32.3 | 7.1 / 6.3 | 17/47 |
| 3 layers (+ closure rule red, + fatigue yellow) | 0.67 | 2.3 s | 27.0 / 32.3 | 8.3 / 6.5 | 42/47 |
| closure rule alone (eyeBlink - baseline > 0.3 for 1.5 s) | 0.42 | 8.8 s | 0 / 0 | 0.6 / 0.6 | 22/47 |
| red from the rule only, yellow from the tank | 0.67 | 2.3 s | 21.6 / **0** | 7.3 / **0.6** | 28/47 |

- The closure rule adds no NTHU events the GRU misses, but it is a clean red trigger (0 false reds on NTHU awake,
  0.6/h on UTA alert). Red from the tank (score 100) gives 32 false reds/h, mostly NTHU005, and red cannot be muted.
- The 60 s fatigue alarm at p > 0.5 flags 42/47 drowsy videos but also 29/48 alert videos, 27% of their time:
  not usable as an alarm at window AUC 0.71. Keep it as context (display, adaptation), not as an alert.
- The last arm was added after seeing the first comparison, and the rule threshold 0.3 was not tuned on inner
  folds: it is a proposal to confirm on new data, not a tested result.
- ONNX step export (`tools/export_gru_onnx.py`, `tools/check_gru_onnx.py`): 53 KB, ONNX Runtime matches PyTorch
  within 3.8e-6, 0.016 ms per frame on the dev Mac (1 thread); not yet measured on the Pi.

### New data (2026-11-20, not yet used for training)
- **Raw sources deleted 2026-11-21** (team lead's decision, to free disk): `data/NTHU`, `data/SUST`, `data/raw/UTA`,
  `data/raw/NTHU`. Only the extracted per-frame CSVs (`data/processed/*`), the cache and `data/kaggle_pack` remain.
  Re-extracting (new face model, new features) needs a fresh download; the NTHU 18-person set's source is unknown.
- `data/NTHU`: NTHU-DDD training set, 18 people, 723,248 square face crops (~223-340 px, size varies frame to frame),
  5 scenarios incl. night IR (005 has no `nightglasses`). The filename suffix is the **head** label (0 normal,
  1 nodding, 2 looking aside; checked frame by frame against the mirror's drowsy labels on 001/002/005/006);
  **no drowsiness label** for the 14 new people. Extracted with `python -m tools.extract_public nthu18 data/NTHU
  data/processed/nthu18`: 356 sessions, 359,156 frames at 12 fps; face found median 0.98-1.0 per scenario, low
  only in some `nonsleepyCombination` (looking aside; worst 0.10, NTHU032 nightglasses).
- Crops vs the mirror's full frames on the same frames (001/002/005): eyeBlink r 0.92-0.99, jawOpen 0.91-1.0,
  eyeLookDown 0.86-0.97, pitch 0.95-1.0, yaw 0.89-0.995, roll 0.89-0.996, angle offsets <= 3 deg (removed by the
  calibration pose). The two sources can be mixed.
- Weak labels if no label files turn up: in the 4 labelled people, drowsy share is 88.5% of `sleepyCombination`,
  71.6% of `slowBlinkWithNodding`, 0.5% of `nonsleepyCombination` frames.
- `data/SUST`: SUST-DDD, 2,074 clips of 10 s (975 drowsy / 1,099 not), 19 drivers in real cars, 720x1280 portrait,
  label per clip by a 3-person vote; no person id in the files. Extracted with `python -m tools.extract_public sust
  data/SUST data/processed/sust` (scaled to 360x640): 2,074 sessions x 120 frames; face found median 1.0, below
  0.8 in 7 drowsy and 49 not-drowsy clips, never 0. Source fps 30 (1,054), 25 (420), 29.97 (240), 29.5 / 29.83 /
  29.92 (60 each): fps groups of exactly 60 clips hint at one phone = one driver.
- SUST person ids (`tools/sust_people.py`, 2026-11-21, **confirmed by the team lead on the face sheets and applied 2026-11-21**): fps runs alone do not
  give people (a person edge can sit 1 clip off an fps edge; one 30 fps run holds 3 people), and scene/face-geometry
  cuts over-split (light changes while driving). Grouping was read off per-clip face sheets: people appear in the
  same order in `d_*` and `n_*`. Result: 19 people (3 women, 16 men, as the README states), all 2,074 clips assigned
  once. Doubts: SUST10 joins two looks (haircut/lighting) that sit next to each other in both series; joining is the
  safe side for leave-one-person-out. SUST16 has 60 drowsy and no not-drowsy clip (no awake baseline). SUST19's
  d_621/626/627 sit inside SUST12's run. Check sheets: `runs/sust_people/check_*.jpg`; `--apply` wrote `subject_id`/`subject_source` into all 2,074 metas (0 missing).
- Loader: `load_all(nthu18=True)` (`load_nthu18`) replaces the 4-person mirror: 267 sequences (yawning excluded);
  21 with frame labels (001/002/005/006 daytime, taken from the mirror by t_ms; targets identical to `load_nthu`),
  246 weak (`weak_target` per sequence, frame `target` NaN, so the current training code ignores them).
- Loader `load_sust` (`load_all(sust=True)`, 2026-11-21): label_kind "clip", frame `target` NaN, `weak_target` per
  clip. Baseline per setup = (person, source fps, `SUST_SETUP_SPLIT`), from the setup's first 3 not-drowsy clips
  (30 s), which are never scored (66 clips). Why per setup, not per person: one person's clips come from more than
  one phone placement. With a per-person baseline, awake yaw after baseline was +39 deg (SUST05) and +37 deg (SUST10).
  SUST05's two placements follow its two fps (+16 / -28 deg raw yaw). SUST10 has a step inside one fps run at d_528 /
  n_555 (-13 to +15 deg), hard-coded in `SUST_SETUP_SPLIT`. After the fix, mean awake pitch/yaw/roll over all scored
  not-drowsy clips is +0.1 / +1.5 / +1.1 deg; drowsy clips: pitch -2.4 deg, eyeBlink +0.05 over baseline. Left as is:
  SUST06's drowsy clips at 29.83 fps have no awake clip at that fps (person baseline, about -10 deg yaw offset);
  SUST16 uses the population baseline (test only); SUST17 has 3 not-drowsy clips left after calibration.
- `tools/pack_kaggle.py --sources nthu18 sust`: nthu18 167 MB, sust 142 MB in `data/kaggle_pack` (uploaded as v6, re-bundled as v7, see the Kaggle queue note below).
  Packed vs local loaders: same 2,341 sessions, max |x| difference 4e-4 (float32), targets/weak/calib identical.

### GRU v2 on the new data, no retraining (2026-11-21, `python -m tools.eval_new_data runs/gru_v2/gru_h48`,
report `runs/gru_v2/new_data.json`)
Each of the 4 fold models is tested on the 14 NTHU people and 19 SUST people that none of them has seen, plus its own
held-out mirror person on the crops. Decision = bench `gru_offset` + tank.

| fold (thr) | NTHU-18 seq AUC | sleepy seqs reach warning | slowBlink reach warning | nonsleepy reach warning | SUST clip AUC (episode / fatigue mean) | SUST drowsy clips reach warning | SUST awake warnings/h |
|---|---|---|---|---|---|---|---|
| 0 (9.9) | 0.968 | 0.26 | 0.27 | 0.00 | 0.575 / 0.535 | 0.02 | 0.7 |
| 1 (7.8) | 0.960 | 0.29 | 0.32 | 0.00 | 0.691 / 0.621 | 0.05 | 0.3 |
| 2 (3.4) | 0.927 | 0.93 | 0.92 | 0.47 | 0.680 / 0.713 | 0.25 | 18.1 |
| 3 (6.4) | 0.933 | 0.65 | 0.72 | 0.03 | 0.618 / 0.563 | 0.03 | 0.7 |

- Crops vs mirror on the held-out person, same model, frame AUC: NTHU001 0.960 vs 0.964, NTHU002 0.963 vs 0.969,
  NTHU005 0.838 vs 0.774, NTHU006 0.861 vs 0.874. The crop set does not hurt the model.
- Ranking carries over to new NTHU people (sequence AUC 0.93-0.97). The alarm threshold does not: the offset threshold
  learnt from 3 training people ranges 3.4-9.9 logits across folds, so one fold alarms on 47% of nonsleepy sequences
  and another reaches warning on only 26% of sleepy ones. That is the part 14 more people should fix.
- SUST (real cars, 10 s clips) is much harder: clip AUC 0.58-0.69, per-person AUC > 0.5 for 13-16 of 18 people;
  SUST10 (0.41-0.49) and SUST12 (0.26-0.46) are below 0.5 in every fold. The tank almost never reaches warning
  inside a 10 s clip, since each clip starts from h = 0 and an empty tank.

### GRU v3 training plan (agreed 2026-11-21, `python -m tools.train_v3`)
- What each source teaches:
  - NTHU framed people (001/002/005/006 daytime): frame labels, as in v2.
  - NTHU-18 weak sequences, per frame: `nonsleepyCombination` frames are awake; `head_aside` frames are awake
    (0.5% drowsy in the framed people); `head_nodding` frames are drowsy (96.8%). Nodding inside nonsleepy is left
    out, and head-label changes get +-0.75 s margins.
  - The rest of the sleepy and slowBlink sequences is a bag (10 s crop, label 1). Nonsleepy sequences are also
    bags with label 0.
  - SUST clips: bags with the clip's vote label.
  - UTA: fatigue head, as in v2.
- Bag loss: mean of the top 50% frame logits vs the bag label. Why not all frames: 12-28% of a sleepy sequence is
  awake, so "every frame drowsy" would teach that a plain blink is drowsy. Why not only nodding: only 16% of drowsy
  frames are nodding.
- Sampling:
  - Episode crops: half from frame-labelled sessions, half from weak frames.
  - Bags: even over (source, label), so the 1,600 SUST bags do not swamp the ~200 NTHU ones.
- Arms (3 seeds each):
  - A: v2 recipe on the new folds.
  - B: A + NTHU-18.
  - C: B + SUST.
  - D: C with `--lam 0`.
  - B0: the rule.
- Folds:
  - 6 outer folds by person. Each framed NTHU person is in its own fold; 3 NTHU people and 3-4 SUST people per fold.
  - UTA official folds are tested in outer folds 0-3.
  - SUST16 is test only (fold 3).
  - Inner 3 folds pick the epoch (mean of the NTHU frame AUC, NTHU-18 sequence AUC, SUST clip AUC and UTA window
    AUC), the temperature, and the offset threshold.
  - The threshold is set at 5% false positives on pooled inner awake frames (target 0, plus not-drowsy SUST clips).
- Code ships inside the kernel (`tools/build_kernel.py`, base64 tar.gz in run.py). The dataset holds data only, so a
  code change does not need a new dataset version.
- Smoke test 2026-11-21 (`--quick`, 1 fold, 10 steps): all five arms run end to end, both locally and through
  run.py on `data/kaggle_pack`. B0 gives the same numbers from the local and the packed data.
- Kaggle queue (2026-11-21): the first v3 kernel stayed QUEUED for over 45 min with dataset v6 (5,216 loose files);
  probe kernels with no dataset (CPU, and GPU, which got a T4) started within a minute, and a GPU probe that mounted v6
  also stayed QUEUED. First guess was the file count, so dataset v7 holds one plain tar per source
  (`tools/bundle_kaggle.py`; named `*.pack` so Kaggle does not unpack them) and run.py unpacks them to
  `/kaggle/temp/data` (6.2 s on Kaggle, 2,484 sessions). The v7 kernel still sat QUEUED, so the file count was not
  the cause. What the runs show: every kernel that named `machine_shape: NvidiaTeslaT4` stayed QUEUED; a GPU kernel
  with no machine_shape got a T4 within a minute with v7 mounted, and so did a CPU kernel. The queued versions also
  hold the account's 2 batch GPU sessions ("Maximum batch GPU session count of 2 reached"), and the CLI cannot cancel
  them. build_kernel.py no longer sets machine_shape or docker_image, and `--cpu` builds a CPU kernel.
- After the queued v3 kernel versions were cancelled on the web, GPU kernels with no machine_shape got
  a T4 at once. Speed: A seed 0 (6 outer folds) took 10.9 min on the T4 and C seed 0 took 15.6 min. On the Kaggle CPU,
  one outer fold of A took about 21 min (about 2.7 h for A). Use the GPU; CPU is the fallback.

### GRU v3 first results (2026-11-22, seed 0 only, reports in `runs/v3/{B0_kaggle,A_s0,C_s0}/report.json`)
A and C ran on the Kaggle GPU, B0 on the Kaggle CPU.
Test = outer folds pooled.

| metric | B0 | A | C |
|---|---|---|---|
| NTHU framed frame AUC | 0.742 | 0.891 | 0.959 |
| NTHU framed event recall | 0.667 | 0.742 | 0.950 |
| NTHU framed detection delay (s) | 14.6 | 7.9 | 1.2 |
| NTHU framed warnings/h | 42.0 | 36.6 | 79.3 |
| NTHU-18 sequence AUC | 0.760 | 0.953 | 0.990 |
| NTHU-18 sleepy / slowBlink reach warning | 0.90 / 0.88 | 0.83 / 0.81 | 0.93 / 0.95 |
| NTHU-18 nonsleepy reach warning | 0.675 | 0.124 | 0.311 |
| NTHU-18 nonsleepy warnings/h | 116.0 | 10.9 | 28.6 |
| SUST clip AUC | 0.722 | 0.666 | 0.733 |
| SUST people AUC > 0.5 | 16/18 | 15/18 | 17/18 |
| SUST drowsy clips reach warning | 0.17 | 0.21 | 0.34 |
| SUST not-drowsy warnings/h | 20.8 | 21.6 | 34.0 |
| UTA fatigue window AUC | - | 0.731 | 0.767 |
| UTA alert warnings/h | 11.2 | 16.2 | 32.0 |

- C ranks best on every source and detects faster (event recall 0.95, delay 1.2 s).
- C also raises about twice as many false warnings as A. Its offset thresholds are lower (2.4-3.0 logits, against
  3.6-5.5 for A), even though both set the threshold at 5% false positives on inner awake frames.
- So the ranking gain is real, but the operating point needs work (threshold rule, or the tank) before C is better
  for the driver. Seed 0 only: B, D and seeds 1-2 come before any conclusion.

### GRU v3, 4 arms x 3 seeds (2026-11-22, `runs/v3/summary_3seeds.json`)
Each cell is the mean +- sd over seeds 0-2. All 12 runs exit 0. Per run on the T4: A 10-11 min, B 15 min, C 15-16
min, D 11-12 min.

| metric | B0 | A | B (+NTHU-18) | C (+SUST) | D (C, lam 0) |
|---|---|---|---|---|---|
| NTHU framed frame AUC | 0.742 | 0.876+-0.014 | 0.949+-0.005 | 0.947+-0.016 | 0.924+-0.005 |
| NTHU framed event recall | 0.667 | 0.658+-0.084 | 0.839+-0.024 | 0.950+-0.000 | 0.853+-0.064 |
| NTHU framed detection delay (s) | 14.6 | 3.6+-3.9 | 5.8+-0.2 | 0.9+-0.4 | 2.3+-2.1 |
| NTHU framed warnings/h | 42.0 | 46.9+-13.4 | 48.6+-13.3 | 90.1+-11.7 | 55.9+-4.2 |
| NTHU-18 sequence AUC | 0.760 | 0.946+-0.006 | 0.990+-0.003 | 0.988+-0.002 | 0.991+-0.003 |
| NTHU-18 sleepy reach warning | 0.902 | 0.826+-0.011 | 0.904+-0.057 | 0.939+-0.008 | 0.952+-0.006 |
| NTHU-18 nonsleepy reach warning | 0.675 | 0.202+-0.077 | 0.212+-0.060 | 0.292+-0.045 | 0.315+-0.031 |
| NTHU-18 nonsleepy warnings/h | 116.0 | 15.7+-4.5 | 13.5+-4.4 | 27.4+-2.0 | 27.2+-2.1 |
| SUST clip AUC | 0.722 | 0.655+-0.015 | 0.650+-0.012 | 0.720+-0.014 | 0.720+-0.003 |
| SUST people AUC > 0.5 | 16/18 | 14-15/18 | 14-15/18 | 17/18 | 17/18 |
| SUST drowsy clips reach warning | 0.173 | 0.219+-0.035 | 0.243+-0.029 | 0.376+-0.035 | 0.350+-0.030 |
| SUST not-drowsy warnings/h | 20.8 | 28.9+-14.9 | 29.4+-12.5 | 41.0+-7.0 | 37.6+-7.2 |
| UTA fatigue window AUC | - | 0.724+-0.011 | 0.758+-0.010 | 0.755+-0.013 | - (no fatigue head) |
| UTA alert warnings/h | 11.2 | 22.8+-5.9 | 14.8+-3.4 | 35.2+-7.0 | 28.7+-7.0 |

- B vs A: NTHU-18 helps on NTHU (frame AUC 0.876 to 0.949, event recall 0.66 to 0.84) and on UTA, not on SUST.
- C vs B: SUST lifts SUST clip AUC from 0.65 to 0.72 and the event recall to 0.95. That only reaches the rule (B0
  0.722), and the warnings per hour almost double on every source.
- D vs C: dropping the fatigue loss changes little on NTHU-18 and SUST. Framed AUC falls to 0.924, and framed warnings
  fall to 56/h.
- Every arm, and B0, raises 15-90 warnings per hour on awake data. The operating point (threshold rule, tank) is now
  the main problem, more than the ranking. It can be studied offline on the saved `fold*_pred.npz`, with no
  retraining.

### Operating point (2026-11-22, `python -m tools.tune_operating_point runs/v3`, `runs/v3/operating_point.json`)
- Rule searched on out-of-fold scores, per model:
  - z = episode logit - driver offset - the fold's threshold;
  - optional mean over the 3 seeds;
  - EMA of z (alpha);
  - drowsy when EMA > delta;
  - tank (inc, dec). Warning 50 and critical 100 stay fixed.
- Grid:
  - alpha {1, 0.5, 0.25, 0.12, 0.06};
  - delta -1 to 5;
  - tank (5,2), (5,3), (5,4), (5,5), (4,4), (3,3).
- Objective: sensitivity = mean of three numbers: NTHU framed event recall, NTHU-18 sleepy+slowBlink sequences
  reaching warning, SUST drowsy clips reaching warning. Maximise it within a budget of false warnings per hour,
  pooled over 12.3 awake hours: NTHU framed awake, NTHU-18 nonsleepy, SUST not-drowsy, UTA alert.
- Honest numbers: leave-one-fold-out. The rule is chosen on 5 outer folds and scored on the 6th.
- Check: the baseline rule (alpha 1, delta 0, 5/2) on C_s0 gives the report's numbers: UTA 32.0 warnings/h, NTHU-18
  nonsleepy 28.4 vs 28.6. The small gaps come from pooling over frames vs averaging per fold.

LOFO results (sensitivity / false warnings per h):

| budget/h | A ens | B ens | C s0 | C s1 | C s2 | C ens | D ens |
|---|---|---|---|---|---|---|---|
| 2 | 0.35 / 2.3 | 0.45 / 2.9 | 0.39 / 2.1 | 0.39 / 2.5 | 0.35 / 2.1 | 0.41 / 2.8 | 0.32 / 1.9 |
| 4 | 0.43 / 4.8 | 0.52 / 6.2 | 0.49 / 4.4 | 0.47 / 4.8 | 0.40 / 4.5 | 0.54 / 4.5 | 0.52 / 4.1 |
| 6 | 0.47 / 6.0 | 0.55 / 7.3 | 0.56 / 5.8 | 0.54 / 5.9 | 0.48 / 6.2 | **0.62 / 5.5** | 0.54 / 5.8 |
| 10 | 0.52 / 10.1 | 0.62 / 9.5 | 0.61 / 9.6 | 0.64 / 8.7 | 0.58 / 11.3 | 0.65 / 8.8 | 0.61 / 10.1 |

- Without the rule (C_s0, alpha 1, delta 0) the score is 0.75 at 33.6 warnings/h. Cutting false warnings costs
  sensitivity steeply.
- The 3-seed ensemble adds about 0.06 sensitivity over one seed at the same budget. The seeds differ more than the
  arms B, C and D do.
- C ens at a budget of 6/h, LOFO:
  - NTHU framed event recall 0.875 (24 events);
  - NTHU-18 sleepy reach warning 0.85;
  - SUST drowsy clips reach warning 0.13;
  - false warnings per hour: NTHU-18 1.5, SUST 2.8, UTA 6.7.
- SUST stays weak. Each 10 s clip starts cold, and many "drowsy" clips look awake. It is the hardest source, and it
  is not a measure of the runtime.

### Final model (decided 2026-11-22)
- **Arm C (framed NTHU + NTHU-18 weak + SUST bags + UTA fatigue), 3 seeds, averaged.**
- **Rule: EMA alpha 0.12 (about 0.7 s), delta +2.0 logits over each model's threshold, tank unchanged (+5 / -2,
  warning 50, critical 100).** This is the rule the 6/h budget picks on all 6 folds pooled.
- Why C over B or D:
  - C has the best LOFO sensitivity at every budget from 4/h up;
  - it is the only arm trained on real-car data (SUST);
  - D gives up the fatigue head (UTA AUC) and gains nothing.
- Why the ensemble: +0.06 sensitivity at the same false-warning budget, for 3 GRU calls per frame (0.045 ms on the
  dev Mac; Pi 4 not measured yet).
- Training: `python -m tools.train_v3 --arm C --final --seed S` on every person except SUST16, which is test only.
  The inner 3 folds still pick the epoch (8 for all three seeds) and the threshold (2.60 / 3.06 / 2.82). Kaggle
  GPU, 3 min per seed on the T4. Checkpoints are in `runs/v3/final/C_s{0,1,2}/final.pt`.
- Export: `python -m tools.export_final runs/v3/final/C_s*/final.pt --out src/models/eye_gru_v3 --alpha 0.12 --delta 2.0`
  writes 3 ONNX step models (53 kB each) and `rule.json`. ONNX matches PyTorch to 3.6e-6.
- Runtime: `src/core/gru_alarm.GruAlarm` (onnxruntime + numpy). Its per-frame decision feeds the existing
  `AlertManager.update(is_distracted=...)`. `main.py` still runs the MLP; wiring is the next step.
- Calibration must be awake driving. The offset is the median logit over the first 30 s with a face; a calibration
  taken while drowsy hides the drowsiness. Checked end to end on NTHU-18, where the people were seen in training, so
  this is a pipeline check, not a score:
  - calibrating on each sleepy sequence's own start: 34% of sleepy sequences reach warning;
  - calibrating on the person's awake sequence first: 95%, with 0% of nonsleepy sequences reaching warning.
- Open:
  - Pi 4 timing;
  - own recordings (only 10 sessions, not used yet);
  - whether 6 false warnings/h is acceptable for the product (the team decides; 2/h costs 0.2 sensitivity).
- Full training runs on Kaggle only, never on the dev laptop.

### Benchmark against other models (2026-11-23, `python -m tools.run_video`)
Purpose: the report compares GRU v3 with other models doing the same job, on the same videos.
```bash
pip install onnxruntime opencv-python-headless "numpy<2"
python -m tools.run_video path/to/*.mp4 --out runs/bench_gru_v3     # --color rgb for colour video
```
- Output: one CSV per video, one row per processed frame (12 fps): `t_s, face, z, ema, drowsy, ms_face, ms_model`,
  plus `summary.json` (face ratio, drowsy ratio, median / p95 ms per frame).
- `drowsy` is the model's raw per-frame decision (EMA > delta), before any alert rule. Compare it with the other
  model's per-frame output against the same labels.
- Rows with an empty `ema` are still calibrating; leave them out. Videos with no face in the first 30 s are skipped.
Conditions for a fair table:
- Same videos, same labels, same metrics for every model.
- Neither model trained on the benchmark set; if one was, say so in the table.
- Every video starts with about 30 s of the driver awake (GRU calibration); clips shorter than about 35 s give
  no GRU result (e.g. SUST's 10 s clips).
- Timing on the same machine for every model (Pi 4 or one laptop). Checked 2026-11-23 on one own session (232 s):
  2,424 frames, face 6.7 ms + model 0.16 ms per frame on the dev Mac; Pi 4 not measured.

### Open decision: what triggers red (parked 2026-11-20, team lead decides later)
- Update 2026-11-23: a proposed answer, with measurements and references (Euro NCAP, EU 2021/1341), is in
  `scoring-water-tank.md` "Proposed rule". The text below is the earlier state.
- Today red (critical, cannot be muted) = water tank at 100, fed by the GRU: 32 false reds/h on NTHU awake frames.
- Proposal: red only from the closure rule (eyes closed >= 1.5 s, delta 0.3; later + head drop, yawn); yellow from
  GRU + tank at 50; 60 s fatigue head as context only (dashboard, adaptation), never an alarm.
- Options: A proposal, B keep tank-driven red, C other. Not written into `scoring-water-tank.md` (stub) and no alert
  code changed until decided. Also pending: final model trained on all 52 people (`--final`), Pi 4 timing.

### Proposed architecture (2026-11-17 plan; the trained v3 model is in "Final model")
- As trained: Linear 15→32 + GELU → GRU 48 → two heads (episode, fatigue), about 12.4k parameters per ONNX
  model; the runtime uses the episode head only.
- Input per frame (12 fps): `eyeBlink`, `eyeSquint`, `eyeWide`, `eyeLookDown` (L/R each), EAR L/R from landmarks,
  head pitch/yaw/roll from the 4x4 matrix, pitch rate, face-found flag (15 values). Eye features minus the
  person's baseline; head angles minus the calibration pose. Face lost: features 0, flag 0, the GRU keeps running.
- Model: Linear 15→32 + GELU → GRU 1 layer, 48 units → Linear 48→1 + sigmoid. About 10k parameters.
- Runtime: ONNX exported as one step `(x_t, h_{t-1}) → (p_t, h_t)`, one call per frame on ONNX Runtime.
- Training: random 10 s crops; per-frame weighted BCE; loss ignored on the first 2 s of a crop, ±0.75 s around
  label changes, `unlabeled` and yawn frames. Augmentation: small noise, frame rate jitter 10–16 fps, dropped
  frames, random baseline offset, left/right swap. AdamW, early stop on the inner fold, temperature scaling.
- Baselines for the report: B0 per-person `eyeBlink` threshold rule; B1 old MLP; B2 GRU without head features
  and `eyeLookDown`; B3 GRU without per-person normalization.
- Optional second model, decided after UTA fold 1: every 60 s, PERCLOS, blink rate, mean blink duration and mean
  GRU probability → logistic regression → fatigue level for the warning layer and the RL context.

### Data findings that shaped this (2026-11-17)
- Own sessions: camera above eye level and subjects looking down → awake `eyeBlink` median 0.43 vs 0.21 on NTHU
  (frontal camera, same pipeline); awake frames above 0.6: 4.7% vs 0.8%. The product camera must sit frontal, at
  or below eye level.
- NTHU Kaggle mirror: only 4 subjects, daytime, 66,521 frames; the official set (36 subjects, night IR) needs a
  signed licence request.
