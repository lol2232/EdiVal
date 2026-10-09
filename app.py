"""Gradio GUI for evaluating a single edit with EdiVal-Agent.

Upload the original image (A), the edited image (B), type the edit prompt,
and press Evaluate. Models are loaded once at startup, so this needs the same
GPU environment as eval.py (see README "Getting Started").

    python app.py                 # http://localhost:7860
    python app.py --share         # public gradio.live link
    python app.py --tp 1          # run the VLM on a single GPU
"""
import argparse
import json
import re
from typing import Dict, List, Optional, Tuple

import gradio as gr

from detector.utils import _make_json_serializable

LOCAL_TASK_TYPES = [
    "subject_replace", "subject_remove", "material_alter", "color_alter",
    "subject_add", "text_change", "position_change", "count_change",
]
GLOBAL_TASK_TYPES = ["background_change"]
TASK_TYPES = LOCAL_TASK_TYPES + GLOBAL_TASK_TYPES
AUTO = "auto"
GENERIC = "generic (VLM judge)"

# Patterns mirror detector.instruction_detector.paraser_instruction; order matters
# (text_change must be tried before subject_replace/subject_add). The last field
# lists which capture groups are objects (used to mask objects for consistency).
TASK_PATTERNS = [
    ("text_change", r"Replace the text '\[([^\]]+)\]' on \[([^\]]+)\] with '\[([^\]]+)\]'", [2]),
    ("text_change", r"Add text '\[([^\]]+)\]' on the image", []),
    ("subject_replace", r"Replace \[([^\]]+)\] with \[([^\]]+)\]", [1, 2]),
    ("subject_remove", r"Remove \[([^\]]+)\]", [1]),
    ("material_alter", r"Change the material of \[([^\]]+)\] to \[([^\]]+)\]", [1]),
    ("color_alter", r"Change the color of \[([^\]]+)\] to \[([^\]]+)\]", [1]),
    ("subject_add", r"Add \[([^\]]+)\] on the \[([^\]]+)\] of \[([^\]]+)\]", [1, 3]),
    ("subject_add", r"Add \[([^\]]+)\]", [1]),
    ("position_change", r"Change the position of \[([^\]]+)\] to \[([^\]]+)\] of \[([^\]]+)\]", [1, 3]),
    ("count_change", r"Change the count of \[([^\]]+)\] to \[([^\]]+)\]", [1]),
    ("background_change", r"Change the background to \[([^\]]+)\]", []),
    ("background_change", r"Change the background to [^,]+, remain the .+ unchanged", []),
]

GENERIC_JUDGE_PROMPT = """
You are an expert at evaluating image editing success.

I will show you two images: an original image and an edited image. The editing instruction is: "{instruction}"

Please analyze whether the edit successfully implements the instruction. Consider:
1. Was the requested change properly applied?
2. Is the edit accurate and complete?
3. Does the edited image look natural and coherent?

Respond with only "Yes" if the edit is successful, or "No" if it is not successful.
"""

PROMPT_EXAMPLES = """\
**Prompt format.** EdiVal's task-specific checks need the bracketed format below. Any other prompt is scored by a generic VLM yes/no judge instead.

| Task | Example |
|---|---|
| subject_add | `Add [dog] on the [left] of [bench]` or `Add [dog]` |
| subject_remove | `Remove [cup]` |
| subject_replace | `Replace [cat] with [rabbit]` |
| color_alter | `Change the color of [car] to [red]` |
| material_alter | `Change the material of [table] to [glass]` |
| text_change | `Replace the text '[OPEN]' on [sign] with '[CLOSED]'` or `Add text '[HELLO]' on the image` |
| position_change | `Change the position of [ball] to [right] of [dog]` |
| count_change | `Change the count of [apple] to [3]` |
| background_change | `Change the background to [beach]` |
"""

MODELS: Dict[str, object] = {}


def load_models(tensor_parallel_size: int) -> None:
    from detector.instruction_detector import load_instruction_model
    from detector.quality_detector import load_quality_model
    from detector.consistency_detector import load_consistency_model

    MODELS["grounding"], MODELS["vlm"] = load_instruction_model(tensor_parallel_size=tensor_parallel_size)
    MODELS["quality"], MODELS["hps"] = load_quality_model()
    MODELS["dinov3"] = load_consistency_model()


def detect_task_type(prompt: str) -> Optional[str]:
    for task_type, pattern, _ in TASK_PATTERNS:
        if re.search(pattern, prompt):
            return task_type
    return None


def edited_objects(prompt: str) -> List[str]:
    """Objects named in the prompt (not colors, positions, counts, ...)."""
    for _, pattern, groups in TASK_PATTERNS:
        match = re.search(pattern, prompt)
        if match:
            return [match.group(g).strip() for g in groups]
    return []


def split_objects(text: str) -> List[str]:
    return [o.strip() for o in re.split(r"[,\n]", text or "") if o.strip()]


def judge_generic(image_a, image_b, instruction: str) -> Tuple[int, str]:
    from detector.instruction_detector import _query_vlm_2_image
    from detector.utils import load_resize_image

    answer = _query_vlm_2_image(
        MODELS["vlm"], load_resize_image(image_a), load_resize_image(image_b),
        GENERIC_JUDGE_PROMPT.format(instruction=instruction),
    )
    score = 1 if answer.strip().startswith("yes") else 0
    return score, f"Generic VLM judge answered: {answer!r}"


def fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def evaluate_pair(image_a, image_b, prompt: str, task_choice: str, unchanged_text: str):
    from detector.instruction_detector import evaluate_instruction_following
    from detector.quality_detector import evaluate_quality
    from detector.consistency_detector import evaluate_consistency

    if image_a is None or image_b is None:
        raise gr.Error("Please provide both Image A (original) and Image B (edited).")
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter the edit prompt.")

    instruction = prompt.replace("[", "").replace("]", "")
    if task_choice == AUTO:
        task_type = detect_task_type(prompt) or GENERIC
    else:
        task_type = task_choice

    # --- Instruction following ---
    if task_type == GENERIC:
        score, reason = judge_generic(image_a, image_b, instruction)
    else:
        # background_change uses the natural-language instruction (as in eval.py).
        score, reason = evaluate_instruction_following(
            image_a, image_b,
            formatted_instruction=prompt,
            instruction=prompt if task_type == "background_change" else instruction,
            task_type=task_type,
            grounding_model=MODELS["grounding"],
            vlm_model=MODELS["vlm"],
            output_reason=True,
        )

    results = {
        "prompt": prompt,
        "task_type": task_type,
        "instruction_following": score,
        "instruction_following_reason": reason,
    }

    # --- Quality of the edited image ---
    quality = evaluate_quality(image_b, MODELS["quality"], prompt=instruction, hps_inferencer=MODELS["hps"])
    results["quality"] = quality

    # --- Consistency (only when the user names objects that should stay unchanged) ---
    unchanged = split_objects(unchanged_text)
    if unchanged:
        all_objects = list(dict.fromkeys(unchanged + edited_objects(prompt)))
        object_result, background_result = evaluate_consistency(
            image_a, image_b, unchanged, all_objects,
            grounding_model=MODELS["grounding"], dinov3_model=MODELS["dinov3"],
        )
        results["consistency"] = {
            "object_dinov3_consistency_mean": object_result["object_dinov3_consistency_mean"],
            "object_l1_consistency_mean": object_result["object_l1_consistency_mean"],
            "background_l1_consistency": background_result["bg_l1_consistency"],
            "background_dinov3_consistency": background_result.get("bg_dinov3_masked_similarity"),
        }

    results = _make_json_serializable(results)
    return render_summary(results), json.dumps(results, indent=2)


def render_summary(r: Dict) -> str:
    verdict = "✅ **PASS**" if r["instruction_following"] == 1 else "❌ **FAIL**"
    lines = [
        f"### Instruction following: {verdict}",
        f"- Task type: `{r['task_type']}`",
        f"- Reason: {r['instruction_following_reason']}",
        "",
        "### Quality (edited image)",
        "| Metric | Value |", "|---|---|",
    ]
    for key in ["plausibility", "aesthetics", "human_preference_score", "sharpness", "contrast", "mean_s", "high_sat_ratio"]:
        lines.append(f"| {key} | {fmt(r['quality'].get(key))} |")
    if "consistency" in r:
        lines += ["", "### Consistency (A → B)", "| Metric | Value |", "|---|---|"]
        for key, value in r["consistency"].items():
            lines.append(f"| {key} | {fmt(value)} |")
    else:
        lines += ["", "_Consistency skipped: list objects that should stay unchanged under Advanced to compute it._"]
    return "\n".join(lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="EdiVal Evaluator") as demo:
        gr.Markdown("# EdiVal single-edit evaluator\nUpload the original and edited image, enter the edit prompt, then press **Evaluate**.")
        with gr.Row():
            image_a = gr.Image(label="Image A (original)", type="pil")
            image_b = gr.Image(label="Image B (edited)", type="pil")
        prompt = gr.Textbox(label="Prompt", placeholder="e.g. Change the color of [car] to [red]", lines=2)
        with gr.Accordion("Advanced", open=False):
            task_choice = gr.Dropdown([AUTO] + TASK_TYPES + [GENERIC], value=AUTO, label="Task type")
            unchanged = gr.Textbox(
                label="Objects that should stay unchanged (comma-separated, optional)",
                placeholder="e.g. dog, bench",
            )
            gr.Markdown(PROMPT_EXAMPLES)
        button = gr.Button("Evaluate", variant="primary")
        summary = gr.Markdown(label="Result")
        with gr.Accordion("Raw JSON", open=False):
            raw = gr.Code(language="json", label="Result JSON")
        button.click(evaluate_pair, [image_a, image_b, prompt, task_choice, unchanged], [summary, raw])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EdiVal single-edit evaluation GUI")
    parser.add_argument("--tp", type=int, default=2, help="tensor parallel size for the Qwen2-VL judge (number of GPUs)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    args = parser.parse_args()

    print("Loading EdiVal models (first run downloads weights from Hugging Face)...")
    load_models(args.tp)
    build_ui().queue(default_concurrency_limit=1).launch(server_name=args.host, server_port=args.port, share=args.share)
