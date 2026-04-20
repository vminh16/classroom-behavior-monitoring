import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
USER_ROOT = ROOT.parent
PADDLE_DET_ROOT = USER_ROOT / "PaddleDetection"
PADDLE_CLAS_ROOT = USER_ROOT / "PaddleClas"

DEFAULT_WORKSPACE_ROOT = ROOT / "output" / "ood_holdout"
DEFAULT_DET_MODEL = ROOT / "output_inference" / "picodet_m_416_classroom"
DEFAULT_CLS_MODEL = ROOT / "output_inference" / "pplcnet_behavior"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_exists(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def format_cmd(cmd):
    return subprocess.list2cmdline([str(part) for part in cmd])


def run_step(title: str, cmd):
    print(f"[run] {title}")
    print(format_cmd(cmd))
    subprocess.run(cmd, check=True)


def workspace_paths(workspace_root: Path, tag: str):
    workspace = workspace_root / tag
    pseudo_root = workspace / "pseudo_labels"
    return {
        "workspace": workspace,
        "images_dir": workspace / "images",
        "pseudo_root": pseudo_root,
        "pseudo_images_dir": pseudo_root / "images",
        "annotations_dir": pseudo_root / "annotations",
        "annotations_v2_dir": pseudo_root / "annotations_v2",
        "crops_dir": workspace / "crops",
        "report_dir": workspace / "reports",
    }


def build_prepare_parser(subparsers):
    parser = subparsers.add_parser(
        "prepare",
        help="Extract representative frames and generate PicoDet pseudo boxes.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video-file", type=Path, default=None)
    group.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--tag", required=True, help="Workspace name for this OOD holdout.")
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--frames-per-video", type=int, default=200)
    parser.add_argument("--skip-start-sec", type=float, default=0.0)
    parser.add_argument("--skip-end-sec", type=float, default=0.0)
    parser.add_argument("--random-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--det-model-dir", type=Path, default=DEFAULT_DET_MODEL)
    parser.add_argument("--det-threshold", type=float, default=0.5)
    parser.add_argument("--det-device", default="GPU")
    parser.add_argument("--det-run-mode", default="paddle")
    parser.add_argument("--det-batch-size", type=int, default=1)
    parser.add_argument("--skip-empty", action="store_true")
    parser.set_defaults(func=cmd_prepare)


def build_finalize_parser(subparsers):
    parser = subparsers.add_parser(
        "finalize",
        help="Crop relabeled boxes, build one OOD test list, and run LCNet eval.",
    )
    parser.add_argument("--tag", required=True, help="Workspace name created by prepare.")
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE_ROOT)
    parser.add_argument("--anno-dir", type=Path, default=None)
    parser.add_argument("--cls-model-dir", type=Path, default=DEFAULT_CLS_MODEL)
    parser.add_argument("--labels", default="normal,using_phone,sleeping")
    parser.add_argument("--class-order", default="normal,using_phone,sleeping")
    parser.add_argument("--min-size", type=int, default=16)
    parser.add_argument("--expand", type=float, default=0.0)
    parser.add_argument("--cls-batch-size", type=int, default=32)
    parser.add_argument("--cls-device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.set_defaults(func=cmd_finalize)


def cmd_prepare(args):
    paths = workspace_paths(args.workspace_root, args.tag)
    ensure_exists(PADDLE_DET_ROOT, "PaddleDetection root")
    ensure_exists(PADDLE_CLAS_ROOT, "PaddleClas root")
    ensure_exists(args.det_model_dir, "Detection model dir")
    if args.video_file is not None:
        ensure_exists(args.video_file, "Video file")
    if args.video_dir is not None:
        ensure_exists(args.video_dir, "Video dir")

    ensure_dir(paths["workspace"])

    extract_script = ensure_exists(
        PADDLE_DET_ROOT / "tools" / "extract_frames_for_labeling.py",
        "Frame extraction script",
    )
    predict_script = ensure_exists(
        PADDLE_DET_ROOT / "tools" / "predict_images_to_voc.py",
        "Pseudo label export script",
    )
    label_script = ensure_exists(
        PADDLE_DET_ROOT / "tools" / "label_pseudo_boxes.py",
        "Pseudo box labeler",
    )

    extract_cmd = [sys.executable, str(extract_script)]
    if args.video_file is not None:
        extract_cmd.extend(["--video_file", str(args.video_file)])
    if args.video_dir is not None:
        extract_cmd.extend(["--video_dir", str(args.video_dir)])
    extract_cmd.extend([
        "--output_dir",
        str(paths["images_dir"]),
        "--frames_per_video",
        str(args.frames_per_video),
        "--skip_start_sec",
        str(args.skip_start_sec),
        "--skip_end_sec",
        str(args.skip_end_sec),
        "--random_ratio",
        str(args.random_ratio),
        "--prefix_mode",
        "filename",
        "--seed",
        str(args.seed),
    ])

    run_step("extract frames", extract_cmd)

    predict_cmd = [
        sys.executable,
        str(predict_script),
        "--model_dir",
        str(args.det_model_dir),
        "--image_dir",
        str(paths["images_dir"]),
        "--output_dir",
        str(paths["pseudo_root"]),
        "--threshold",
        str(args.det_threshold),
        "--device",
        args.det_device,
        "--run_mode",
        args.det_run_mode,
        "--batch_size",
        str(args.det_batch_size),
    ]
    if args.skip_empty:
        predict_cmd.append("--skip_empty")

    run_step("predict PicoDet boxes to VOC", predict_cmd)

    print("")
    print("Prepare complete.")
    print(f"Workspace      : {paths['workspace']}")
    print(f"Pseudo images  : {paths['pseudo_images_dir']}")
    print(f"Annotations    : {paths['annotations_dir']}")
    print("Next step: relabel each bbox from 'person' to behavior classes.")
    print("Run:")
    print(
        format_cmd([
            sys.executable,
            str(label_script),
            "--images_dir",
            str(paths["pseudo_images_dir"]),
            "--annotations_dir",
            str(paths["annotations_dir"]),
            "--output_dir",
            str(paths["annotations_v2_dir"]),
        ]))


def cmd_finalize(args):
    paths = workspace_paths(args.workspace_root, args.tag)
    ensure_exists(PADDLE_DET_ROOT, "PaddleDetection root")
    ensure_exists(PADDLE_CLAS_ROOT, "PaddleClas root")
    ensure_exists(args.cls_model_dir, "Classification model dir")
    ensure_exists(paths["pseudo_images_dir"], "Pseudo image dir")

    crop_script = ensure_exists(
        PADDLE_CLAS_ROOT / "tools" / "crop_pseudo_labels.py",
        "Crop pseudo labels script",
    )
    build_list_script = ensure_exists(
        PADDLE_CLAS_ROOT / "tools" / "build_classification_list.py",
        "Classification list builder",
    )
    eval_script = ensure_exists(
        PADDLE_CLAS_ROOT / "tools" / "lcnet_eval_report.py",
        "LCNet eval script",
    )

    anno_dir = args.anno_dir or paths["annotations_v2_dir"]
    ensure_exists(anno_dir, "Relabeled annotation dir")

    ensure_dir(paths["crops_dir"])
    ensure_dir(paths["report_dir"])

    crop_cmd = [
        sys.executable,
        str(crop_script),
        "--image-dir",
        str(paths["pseudo_images_dir"]),
        "--anno-dir",
        str(anno_dir),
        "--output-dir",
        str(paths["crops_dir"]),
        "--min-size",
        str(args.min_size),
        "--expand",
        str(args.expand),
    ]
    if args.labels.strip():
        crop_cmd.extend(["--labels", args.labels])

    run_step("crop relabeled boxes", crop_cmd)

    test_list = paths["crops_dir"] / "ood_test_list.txt"
    label_list = paths["crops_dir"] / "label_list.txt"
    build_list_cmd = [
        sys.executable,
        str(build_list_script),
        "--dataset-root",
        str(paths["crops_dir"]),
        "--output-list",
        str(test_list),
        "--label-list",
        str(label_list),
        "--class-order",
        args.class_order,
    ]
    run_step("build OOD test list", build_list_cmd)

    report_path = paths["report_dir"] / "ood_eval_report.txt"
    errors_csv = paths["report_dir"] / "ood_eval_errors.csv"
    eval_cmd = [
        sys.executable,
        str(eval_script),
        "--model_dir",
        str(args.cls_model_dir),
        "--val_list",
        str(test_list),
        "--image_root",
        str(paths["crops_dir"]),
        "--label_list",
        str(label_list),
        "--batch_size",
        str(args.cls_batch_size),
        "--device",
        args.cls_device,
        "--gpu_id",
        str(args.gpu_id),
        "--report",
        str(report_path),
        "--errors_csv",
        str(errors_csv),
    ]
    run_step("run LCNet OOD evaluation", eval_cmd)

    print("")
    print("Finalize complete.")
    print(f"Crops         : {paths['crops_dir']}")
    print(f"Test list     : {test_list}")
    print(f"Label list    : {label_list}")
    print(f"Report        : {report_path}")
    print(f"Errors CSV    : {errors_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Two-phase OOD holdout pipeline for PicoDet + LCNet evaluation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_prepare_parser(subparsers)
    build_finalize_parser(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
