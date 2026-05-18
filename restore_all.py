r"""Batch NAFNet full-res deblur across ALL sources, then stage + COLMAP.

Comprehensive "test everything" sweep (see the approved plan): deblur
every photo dataset AND the 4K videos at FULL resolution, stage through
the normal pipeline, and run COLMAP — so a later training A/B (deblurred
vs every memory.md baseline) is one step away once the RX 7900 XTX
returns. Training itself is OUT OF SCOPE here (VkSplat needs the removed
7900 XTX; the 1070 Ti is inference-only).

This is a THIN orchestrator. It does not reimplement deblur / EXIF /
resize / SfM — it shells out to the proven scripts:
  restore_frames.py   — NAFNet GPU deblur (+ --exif-source/--full-out/
                         --downscale-width added in Part A)
  prepare_dataset.py   — EXIF-orient + INTER_AREA tiers (Class 2/4)
  run_colmap.py / run_colmap_sequential.py — SfM (Class 2/4)

MUST be run with the NVIDIA CUDA env (only torch that sees the 1070 Ti):
  E:\nvidia-gsplat\python312\python.exe E:\vksplat\restore_all.py

Sources are NEVER modified. All output goes to new *_restored dirs /
new dataset folders. Idempotent: a job whose outputs already exist is
skipped, so an interrupted run resumes safely. Continue-on-error: one
job failing does not abort the rest.

Usage:
  ...python.exe restore_all.py                       # everything
  ...python.exe restore_all.py --dry-run --datasets kitchen
  ...python.exe restore_all.py --datasets sharp,bedroom,full2
  ...python.exe restore_all.py --list                # show job table
"""

import argparse
import glob
import os
import subprocess
import sys
import time

VKSPLAT = os.path.dirname(os.path.abspath(__file__))
DATA = r"E:\vksplat_data"
DL = r"E:\Downloads"
SUMMARY = os.path.join(DATA, "_restore_summary.txt")
LOCK = os.path.join(DATA, "_restore_all.lock")


def _acquire_lock():
    """Single-instance guard. Two runs both loading NAFNet on the one
    1070 Ti crash the CUDA context (0xC0000409, observed). The lock file
    holds the owning PID; a stale lock (PID no longer alive) is broken
    automatically so an orphaned/killed run never blocks forever.
    """
    if os.path.exists(LOCK):
        try:
            with open(LOCK) as f:
                old = int(f.read().strip())
        except Exception:
            old = None
        alive = False
        if old is not None:
            try:
                import ctypes
                h = ctypes.windll.kernel32.OpenProcess(0x1000, 0, old)
                if h:
                    ctypes.windll.kernel32.CloseHandle(h)
                    alive = True
            except Exception:
                alive = False
        if alive:
            print(f"ANOTHER restore_all is running (PID {old}). "
                  f"Refusing to start a second GPU job — it would crash "
                  f"the shared CUDA context. Wait for it or kill PID "
                  f"{old}, then retry.", file=sys.stderr)
            sys.exit(3)
        print(f"Breaking stale lock (PID {old} not alive).")
    with open(LOCK, "w") as f:
        f.write(str(os.getpid()))


def _release_lock():
    try:
        os.remove(LOCK)
    except OSError:
        pass

# Samsung phone primed intrinsics for 2000x924 images_2 (memory.md
# issue #10 — COLMAP degenerates without priming on this camera).
SAMSUNG_CAM = "1343.6,1000,462,0.051"


def _img_width(d):
    """Width of the first JPEG in dir d, or 0 if none/missing."""
    fs = sorted(glob.glob(os.path.join(d, "*.jpg")) +
                glob.glob(os.path.join(d, "*.JPG")))
    if not fs:
        return 0
    import cv2
    im = cv2.imread(fs[0])
    return 0 if im is None else im.shape[1]


def _count(d):
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.lower().endswith(".jpg")])


# --------------------------------------------------------------------------
# Job table. Each job is a dict describing one (source -> outputs) unit.
# kind ∈ {class1_local, class1_src, class2_new, class3_inplace, class4_video}
# --------------------------------------------------------------------------
def build_jobs():
    jobs = []

    # Class 1 — staged datasets WITH existing COLMAP. Restore from the
    # best available res; keep existing sparse/ untouched. Output:
    #   <ds>\images_restored\   (full-res, EXIF baked)
    #   <ds>\images_4_restored\ (downscaled to the dataset's images_4 width)
    # 'local': source is the dataset's own already-oriented images/ tier.
    # 'src':   no local full-res — restore the raw E:\Downloads source
    #          (needs --exif-source); stems are 1:1 with images_4 (verified).
    class1 = [
        ("livingroom",   "local", None),
        ("kitchen",      "local", None),
        ("photos",       "local", None),
        ("frames",       "local", None),
        ("sharp",        "src",   "new fotos"),
        ("bedroom",      "src",   "bedroom normal camera"),
        ("lr_full",      "src",   "living room normal camera fulldataset"),
        ("full_dataset", "src",   "50mb full dataset"),
    ]
    for name, mode, srcname in class1:
        ds = os.path.join(DATA, name)
        i4 = os.path.join(ds, "images_4")
        jobs.append(dict(
            name=name, kind="class1_" + mode,
            src=(os.path.join(ds, "images") if mode == "local"
                 else os.path.join(DL, srcname)),
            exif=(mode == "src"),
            full_out=os.path.join(ds, "images_restored"),
            ds_out=os.path.join(ds, "images_4_restored"),
            ds_width=i4,           # width is read from this dir at run time
        ))

    # Class 2 — unstaged photo dumps (memory.md-rejected as blurry). NO
    # tiers/COLMAP. Full pipeline: restore full-res -> prepare_dataset
    # tiers -> COLMAP on the (deblurred) images_2. Poses derive from
    # restored pixels — documented caveat (no clean original was staged).
    class2 = [
        ("full2",           "new fotos full dataset"),
        ("locked_shutter",  "new locked shutter auto iso"),
        ("bedroom_1614",    "bedroom 16-14"),
        ("bedroom_1615_mf", "bedroom 16-15 multifocus"),
    ]
    for name, srcname in class2:
        ds = os.path.join(DATA, name)
        jobs.append(dict(
            name=name, kind="class2_new",
            src=os.path.join(DL, srcname), exif=True,
            full_out=os.path.join(ds, "images_restored"),
            ds_root=ds,
        ))

    # Class 3 — combined photos+video, COLMAP sparse/1_clean exists.
    # Restore the 3 subdirs in place (2000-wide, already train-res).
    comb = os.path.join(DATA, "livingroom_combined")
    for sub in ("photos", "video1", "video2"):
        jobs.append(dict(
            name=f"combined/{sub}", kind="class3_inplace",
            src=os.path.join(comb, sub), exif=False,
            full_out=os.path.join(comb, sub + "_restored"),
        ))

    # Class 4 — 4K videos. Extract ALL frames (no sharp pre-filter,
    # per user) -> deblur full-res -> stage one dataset per video ->
    # sequential COLMAP (video is sequential).
    for vid in sorted(glob.glob(os.path.join(DL, "20260515_*.mp4"))):
        stem = os.path.splitext(os.path.basename(vid))[0]
        ds = os.path.join(DATA, "vid_" + stem)
        jobs.append(dict(
            name="vid_" + stem, kind="class4_video",
            src=vid, exif=False,
            ds_root=ds,
            work=os.path.join(ds, "_extracted"),
        ))
    return jobs


# --------------------------------------------------------------------------
# Sub-process helpers (all use sys.executable = the NVIDIA env we run in).
# --------------------------------------------------------------------------
def run(cmd, log):
    """Run a child script, streaming nothing but capturing for the log.
    Returns (rc, tail-of-output)."""
    log.write(f"\n$ {' '.join(cmd)}\n")
    log.flush()
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    log.write(out)
    log.flush()
    tail = "\n".join(out.strip().splitlines()[-6:])
    return p.returncode, tail


def restore(src, full_out, exif, log, ds_out=None, ds_width=0,
            dry=False, sample=12):
    cmd = [sys.executable, os.path.join(VKSPLAT, "restore_frames.py"),
           src, "--method", "dl", "--out-dir", full_out]
    if exif:
        cmd.append("--exif-source")
    cmd += ["--full-out", full_out]
    if ds_out and ds_width > 0:
        cmd += ["--downscale-width", str(ds_width),
                "--downscale-out", ds_out]
    if dry:
        cmd += ["--dry-run", "--sample", str(sample)]
    return run(cmd, log)


def prepare(src_full, ds_root, log):
    # src_full is already EXIF-oriented (restore baked it) -> tiers only.
    cmd = [sys.executable, os.path.join(VKSPLAT, "prepare_dataset.py"),
           src_full, ds_root, "--skip-fullres"]
    return run(cmd, log)


def colmap(ds_root, sequential, log):
    script = ("run_colmap_sequential.py" if sequential
              else "run_colmap.py")
    cmd = [sys.executable, os.path.join(VKSPLAT, script),
           ds_root, "--image-subdir", "images_2"]
    if not sequential:
        cmd += ["--camera-params", SAMSUNG_CAM, "--no-refine-extra"]
    return run(cmd, log)


def extract_frames(video, work, log):
    """Reuse select_sharp_frames.extract_video_candidates (ALL frames at
    5 fps, no bucketed selection).

    Writes a `_extract_done` sentinel only on full success so a cached
    `work` dir is trusted only if the prior extraction completed — an
    interrupted extraction (killed mid-ffmpeg) re-extracts rather than
    silently using a truncated frame set.
    """
    sys.path.insert(0, VKSPLAT)
    from select_sharp_frames import extract_video_candidates
    log.write(f"\n# ffmpeg extract ALL frames @5fps: {video}\n")
    log.flush()
    files = extract_video_candidates(video, work, fps=5.0,
                                     start=0.0, duration=0.0)
    with open(os.path.join(work, "_extract_done"), "w") as f:
        f.write(str(len(files)))
    log.write(f"  extracted {len(files)} frames -> {work}\n")
    log.flush()
    return files


# --------------------------------------------------------------------------
# Per-job driver.
# --------------------------------------------------------------------------
def do_job(j, log, dry):
    name, kind, src = j["name"], j["kind"], j["src"]
    print(f"\n=== {name}  [{kind}] ===")
    if not os.path.exists(src):
        print(f"  SKIP: source missing: {src}")
        return dict(name=name, status="skip-no-src")

    t0 = time.time()

    if kind.startswith("class1") or kind == "class3_inplace":
        full_out = j["full_out"]
        ds_out = j.get("ds_out")
        ds_width = _img_width(j["ds_width"]) if j.get("ds_width") else 0
        nsrc = _count(src)
        # Idempotent: complete if full_out has all frames (+ ds_out if used).
        done = (_count(full_out) >= nsrc and nsrc > 0 and
                (not ds_out or _count(ds_out) >= nsrc))
        if done and not dry:
            print(f"  SKIP: already complete ({nsrc} frames)")
            return dict(name=name, status="skip-done", n=nsrc)
        rc, tail = restore(src, full_out, j["exif"], log,
                           ds_out=ds_out, ds_width=ds_width, dry=dry)
        print("  " + tail.replace("\n", "\n  "))
        return dict(name=name, status=("dry" if dry else
                    ("ok" if rc == 0 else f"FAIL rc={rc}")),
                    n=nsrc, secs=int(time.time() - t0))

    if kind == "class2_new":
        ds_root, full_out = j["ds_root"], j["full_out"]
        nsrc = _count(src)
        if dry:
            rc, tail = restore(src, full_out, j["exif"], log, dry=True)
            print("  " + tail.replace("\n", "\n  "))
            return dict(name=name, status="dry", n=nsrc)
        # 1) full-res deblur (EXIF baked)
        if _count(full_out) < nsrc:
            rc, tail = restore(src, full_out, j["exif"], log)
            if rc != 0:
                print(f"  FAIL restore rc={rc}\n  " + tail)
                return dict(name=name, status="FAIL-restore", n=nsrc)
        else:
            print(f"  restore already done ({nsrc})")
        # 2) prepare_dataset tiers from the restored full-res
        if _count(os.path.join(ds_root, "images_2")) < nsrc:
            rc, tail = prepare(full_out, ds_root, log)
            if rc != 0:
                print(f"  FAIL prepare rc={rc}\n  " + tail)
                return dict(name=name, status="FAIL-prepare", n=nsrc)
        else:
            print("  tiers already staged")
        # 3) COLMAP on images_2 (exhaustive + primed; memory.md #10)
        if not os.path.isdir(os.path.join(ds_root, "sparse", "0")):
            rc, tail = colmap(ds_root, sequential=False, log=log)
            status = "ok" if rc == 0 else f"FAIL-colmap rc={rc}"
        else:
            print("  COLMAP already done")
            status = "ok"
        print("  " + tail.replace("\n", "\n  "))
        return dict(name=name, status=status, n=nsrc,
                    secs=int(time.time() - t0))

    if kind == "class4_video":
        ds_root, work = j["ds_root"], j["work"]
        if dry:
            print("  (dry-run: video extraction skipped)")
            return dict(name=name, status="dry")
        os.makedirs(ds_root, exist_ok=True)
        # 1) extract ALL frames. Trust a cached work dir ONLY if the
        # prior extraction completed (sentinel present); otherwise a
        # killed-mid-ffmpeg run would feed a truncated frame set.
        done_marker = os.path.join(work, "_extract_done")
        frames = (sorted(glob.glob(os.path.join(work, "cand_*.jpg")))
                  if os.path.isfile(done_marker) else [])
        if not frames:
            frames = extract_frames(src, work, log)
        nfr = len(frames)
        full_out = os.path.join(ds_root, "images_restored")
        # 2) full-res deblur every extracted frame (ffmpeg out: no EXIF)
        if _count(full_out) < nfr:
            rc, tail = restore(work, full_out, exif=False, log=log)
            if rc != 0:
                print(f"  FAIL restore rc={rc}\n  " + tail)
                return dict(name=name, status="FAIL-restore", n=nfr)
        # 3) tiers + 4) sequential COLMAP
        if _count(os.path.join(ds_root, "images_2")) < nfr:
            rc, tail = prepare(full_out, ds_root, log)
            if rc != 0:
                return dict(name=name, status="FAIL-prepare", n=nfr)
        if not os.path.isdir(os.path.join(ds_root, "sparse", "0")):
            rc, tail = colmap(ds_root, sequential=True, log=log)
            status = "ok" if rc == 0 else f"FAIL-colmap rc={rc}"
        else:
            status = "ok"
        return dict(name=name, status=status, n=nfr,
                    secs=int(time.time() - t0))

    return dict(name=name, status="unknown-kind")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=None,
                    help="Comma list to restrict (job names, e.g. "
                         "kitchen,sharp,full2,combined/video1)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Per-job restore_frames --dry-run (sharpness "
                         "report only; no writes / no COLMAP)")
    ap.add_argument("--list", action="store_true",
                    help="Print the job table and exit")
    args = ap.parse_args()

    jobs = build_jobs()
    if args.datasets:
        want = {x.strip() for x in args.datasets.split(",")}
        jobs = [j for j in jobs if j["name"] in want]
        if not jobs:
            print("No jobs match --datasets", file=sys.stderr)
            sys.exit(1)

    if args.list:
        for j in jobs:
            print(f"{j['name']:<24} {j['kind']:<16} src={j['src']}")
        return

    # GPU single-instance guard (every path below loads NAFNet, incl.
    # --dry-run). Two concurrent runs crash the shared CUDA context.
    _acquire_lock()
    try:
        print(f"{len(jobs)} jobs. Summary -> {SUMMARY}")
        results = []
        with open(SUMMARY, "a", encoding="utf-8") as log:
            log.write(f"\n\n===== restore_all run "
                      f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                      f"(dry={args.dry_run}) =====\n")
            for j in jobs:
                try:
                    results.append(do_job(j, log, args.dry_run))
                except Exception as e:
                    print(f"  EXCEPTION in {j['name']}: {e}")
                    results.append(dict(name=j["name"],
                                        status=f"EXC {type(e).__name__}"))
                log.flush()

            log.write("\n--- consolidated ---\n")
            hdr = f"{'job':<24} {'status':<16} {'N':>6} {'secs':>7}\n"
            log.write(hdr)
            print("\n" + hdr.rstrip())
            for r in results:
                line = (f"{r['name']:<24} {r.get('status',''):<16} "
                        f"{r.get('n',''):>6} {r.get('secs',''):>7}")
                log.write(line + "\n")
                print(line)
        print(f"\nFull log: {SUMMARY}")
    finally:
        _release_lock()


if __name__ == "__main__":
    main()
