"""Publish controlled visual ablations and splice their table into main.tex."""
from __future__ import annotations

import json
import os
import re
import tempfile


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUN_RESULTS = os.path.join(ROOT, "04_Experiments", "runs", "visual_ablation", "results.json")
BASELINES = os.path.join(HERE, "results", "tuned_baselines.json")
SUMMARY = os.path.join(HERE, "results", "visual_ablation.json")
MAIN_TEX = os.path.join(HERE, "main.tex")

ROWS = (
    ("dino_small_center_cls", "DINOv2-S thumbnail (CLS)"),
    ("dino_small_center_mean", "DINOv2-S thumbnail (patch mean)"),
    ("dino_small_fit_cls", "DINOv2-S full thumbnail (CLS)"),
    ("dino_base_center_cls", "DINOv2-B thumbnail (CLS)"),
    ("clip_base_center", "CLIP ViT-B/32 thumbnail"),
    ("resnet50_center", "ResNet-50 thumbnail"),
    ("frames_dino_small_mean", "DINOv2-S mean of 20 frames"),
)


def _atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".visual_summary_", suffix=".json", dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def compact(full: dict, baselines: dict) -> dict:
    """Retain provenance, seed values and paired structured comparisons."""
    if full.get("seeds") != [0, 1, 2, 3, 4]:
        raise ValueError("the report requires visual-ablation seeds 0..4")
    if any(run.get("test_evaluated") is not False for run in full["runs"]):
        raise ValueError("refusing to publish a run that evaluated test")
    structured = baselines["feature_sets"]["meta+sched"]["aggregate"]["xgboost"]["auc_roc"]
    selected = full["aggregate"]["thumbnail_frames_text_fields_meta_sched"][
        "late_fusion_mlp"
    ]["auc_roc"]
    differences = [value - reference for value, reference in zip(
        selected["values"], structured["values"],
    )]
    return {
        "generated_at_utc": full["generated_at_utc"],
        "protocol": full["protocol"],
        "n_labelled": full["n_labelled"],
        "seeds": full["seeds"],
        "variants": full["variants"],
        "embedding_provenance": full["embedding_provenance"],
        "aggregate": full["aggregate"],
        "comparison": {
            "selected_model": "thumbnail_frames_text_fields_meta_sched/late_fusion_mlp",
            "tuned_metadata_schedule_xgboost": structured,
            "selected_minus_structured_mean": sum(differences) / len(differences),
            "selected_wins": sum(value > 0 for value in differences),
        },
    }


def render(summary: dict) -> str:
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{l cc}", "\\toprule",
        "Frozen visual representation & Linear & MLP \\\\", "\\midrule",
    ]
    for key, label in ROWS:
        models = summary["aggregate"][key]
        cells = []
        for model in ("linear_probe", "late_fusion_mlp"):
            auc = models[model]["auc_roc"]
            cells.append(f"{auc['mean']:.3f} \\tiny{{$\\pm$ {auc['std']:.3f}}}")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}",
        "\\caption{Visual-only validation AUC-ROC over the same five channel-grouped "
        "splits. The controlled changes isolate encoder size, framing, pooling and "
        "pretraining; frame vectors are averaged over the 20 stored samples.}",
        "\\label{tab:visual-ablation}", "\\end{table}",
    ]
    aggregate = summary["aggregate"]
    small = aggregate["dino_small_center_cls"]["linear_probe"]["auc_roc"]
    base = aggregate["dino_base_center_cls"]["linear_probe"]["auc_roc"]
    clip = aggregate["clip_base_center"]["linear_probe"]["auc_roc"]
    frames = aggregate["frames_dino_small_mean"]["late_fusion_mlp"]["auc_roc"]
    multimodal = aggregate["thumbnail_frames_text_fields_meta_sched"]["late_fusion_mlp"]["auc_roc"]
    comparison = summary["comparison"]
    structured = comparison["tuned_metadata_schedule_xgboost"]
    lines += [
        "",
        f"Increasing DINOv2 from ViT-S to ViT-B changes the linear-probe AUC from "
        f"${small['mean']:.3f}$ to ${base['mean']:.3f}$; model capacity therefore does "
        "not explain the weak thumbnail result. CLIP is the strongest thumbnail-only "
        f"linear representation at ${clip['mean']:.3f}\\pm{clip['std']:.3f}$, whereas "
        f"mean-pooled frame features reach ${frames['mean']:.3f}\\pm{frames['std']:.3f}$ "
        "with the MLP. Adding both thumbnails and frames to field-aware text and metadata "
        f"reaches ${multimodal['mean']:.3f}\\pm{multimodal['std']:.3f}$, still "
        f"${abs(comparison['selected_minus_structured_mean']):.3f}$ below tuned "
        f"metadata+schedule XGBoost (${structured['mean']:.3f}\\pm{structured['std']:.3f}$) "
        f"and higher on {comparison['selected_wins']}/5 matched splits.",
    ]
    return "\n".join(lines)


def splice(document: str, body: str) -> str:
    begin = "%% <<<BEGIN GENERATED visual-ablation>>>"
    end = "%% <<<END GENERATED visual-ablation>>>"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(document):
        raise ValueError("visual-ablation generated markers not found")
    return pattern.sub(lambda _match: f"{begin}\n{body}\n{end}", document)


def main() -> None:
    with open(RUN_RESULTS, encoding="utf-8") as stream:
        full = json.load(stream)
    with open(BASELINES, encoding="utf-8") as stream:
        baselines = json.load(stream)
    summary = compact(full, baselines)
    _atomic_json(SUMMARY, summary)
    with open(MAIN_TEX, encoding="utf-8") as stream:
        document = stream.read()
    with open(MAIN_TEX + ".tmp", "w", encoding="utf-8") as stream:
        stream.write(splice(document, render(summary)))
    os.replace(MAIN_TEX + ".tmp", MAIN_TEX)
    print(f"saved {SUMMARY}")
    print(f"updated {MAIN_TEX}")


if __name__ == "__main__":
    main()
