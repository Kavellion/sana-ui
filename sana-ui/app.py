"""Sana 1.6B Gradio UI -- text-to-image, standalone.

Same shape as Kandinsky UI: no Jupyter, no Colab. Run it, a page
opens in your browser, type a prompt, hit Generate, a picture
appears underneath.

How to run (on a Windows PC with an NVIDIA GPU):

    install.bat      (once -- builds a venv, installs torch + packages)
    run.bat          (starts the app)

The app file lives in the sana-ui/ subfolder; run.bat runs it from
the folder above, so pictures land in images_out/ next to run.bat --
one subfolder per batch name, reused across runs, frames numbered
continuously, and settings.txt appended once per run, exactly like
Kandinsky UI.

First Generate takes a while: the Sana model downloads from
Hugging Face (about 10 GB of bf16 weights) and loads once, then
stays loaded. Repeat generations reuse what's already loaded.

How it works (Diffusers 0.39.0 -- SanaPipeline is in that release,
and its call arguments here match its 0.39.0 signature):
  1. The Gemma2-2B-IT text encoder turns the prompt into embeddings.
  2. The Sana transformer (a 1.6B linear diffusion transformer)
     denoises a latent in a few steps (default 20, DPMSolver
     scheduler).
  3. The DC-AE decoder (32x compression) turns the latent into the
     picture.
Everything runs in bf16 on CUDA, following the model card's own
usage snippet: variant="bf16", torch_dtype=bfloat16, and the VAE
and text encoder moved to bf16 after loading.

Notes that matter for this app:
  - The model repo (Efficient-Large-Model/Sana_1600M_1024px_BF16_
    diffusers) is public and NOT gated: no Hugging Face account,
    no token, no license form. The Gemma weights ship inside that
    repo's text_encoder folder, so nothing is fetched from Google.
  - Sana's prompt budget is 300 tokens (max_sequence_length), not
    CLIP's 77 -- long prompts are welcome; past 300 tokens the tail
    is truncated.
  - Sana generates at the nearest supported ratio internally and
    resizes back to the requested size, so output dimensions match
    the width/height boxes.

Extras, kept in the same plain style as the rest of the app:
size presets, several pictures per click, seed shown in the status
line, a Stop button, a Random prompt button (word banks in code --
no list file to maintain), a session-only prompt history dropdown,
Star batch (copies a batch into images_out/favorites/), a prompt
queue (one prompt per line), and single-level dynamic prompts
{a|b|c}. No animation, no upscaling. Image to image, inpainting,
prompt blending, and ControlNet are NOT in this version -- noted
as follow-ups in the README, not built here.
"""

import os
import random
import re
import shutil
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)  # folder holding run.bat
IMAGES_DIR = os.path.join(ROOT_DIR, "images_out")
FAVORITES_DIR = os.path.join(IMAGES_DIR, "favorites")

MODEL_ID = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"

DEFAULT_PROMPT = (
    "A lighthouse on a rocky cliff during a storm, waves crashing, "
    "dramatic sky, detailed"
)
DEFAULT_NEGATIVE = "low quality, bad quality"

# Random-prompt word banks, in code on purpose: no external list file
# to maintain, and nothing to keep in sync with the app. Plain
# text-to-image subjects only, so any combination reads as a prompt.
_RANDOM_SUBJECTS = [
    "lighthouse", "old castle", "sailing ship", "mountain cabin",
    "red fox", "steam train", "hot air balloon", "medieval knight",
    "space station", "treehouse", "windmill", "dragon",
    "robot explorer", "harbor town", "desert oasis", "snowy owl",
    "vintage car", "floating island",
]
_RANDOM_PLACES = [
    "on a rocky cliff", "in a misty forest", "beside a quiet lake",
    "above the clouds", "in a desert canyon", "on a snowy mountain pass",
    "in a neon city street", "under the sea", "in a medieval village",
    "on a windswept beach", "inside a grand library",
    "at the edge of a volcano", "in a flower meadow",
    "on a frozen river", "in a hidden valley", "above a waterfall",
]
_RANDOM_STYLES = [
    "oil painting", "watercolor", "detailed digital art",
    "cinematic photo", "storybook illustration", "cyberpunk art",
    "art deco poster", "impressionist painting",
    "fantasy concept art", "steampunk illustration", "anime style",
    "vintage photograph", "pixel art", "surrealist painting",
]
_RANDOM_LIGHTING = [
    "dramatic storm light", "golden hour glow", "soft moonlight",
    "misty morning light", "neon glow at night", "warm sunset colors",
    "cold blue twilight", "dappled sunlight", "eerie fog",
    "bright midday sun", "candlelit warmth", "aurora in the sky",
]

_pipe = None  # loaded once on first Generate


def _load_pipe():
    """Load SanaPipeline in bf16 onto CUDA, following the model
    card's own usage snippet. Raises a plain RuntimeError with a
    readable message if there is no CUDA GPU."""
    global _pipe
    if _pipe is not None:
        return _pipe
    import torch
    from diffusers import SanaPipeline

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU found. Sana UI needs an NVIDIA GPU with "
            "a recent driver -- CPU-only mode is not supported."
        )
    _pipe = SanaPipeline.from_pretrained(
        MODEL_ID, variant="bf16", torch_dtype=torch.bfloat16
    )
    _pipe.to("cuda")
    _pipe.vae.to(torch.bfloat16)
    _pipe.text_encoder.to(torch.bfloat16)
    return _pipe


def _safe_batch_name(name):
    """Strip characters Windows forbids in folder names. Falls back to
    a timestamped name when the box is empty."""
    name = (name or "").strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip(" .")
    if not name:
        name = datetime.now().strftime("sana-%Y%m%d-%H%M%S")
    return name


def _unique_folder(base, root=IMAGES_DIR):
    """<root>/<base>, or <base>_2, <base>_3... if it already exists.
    Used only by Star batch now; generations reuse one folder per
    batch name (see _batch_folder) instead of numbered folders."""
    folder = os.path.join(root, base)
    if not os.path.exists(folder):
        return folder
    i = 2
    while True:
        candidate = "{}_{}".format(folder, i)
        if not os.path.exists(candidate):
            return candidate
        i += 1


def _batch_folder(base):
    """The one folder for a batch name: images_out/<base>, created
    if missing, reused if it already exists. Repeat runs never get
    a _2/_3 folder -- frames just keep numbering upward."""
    return os.path.join(IMAGES_DIR, base)


_FRAME_RE = re.compile(r"^(\d{6})\.png$")


def _next_frame_number(folder):
    """First unused frame number in a folder: one past the highest
    existing 000001-style frame, or 1 for a fresh folder."""
    highest = 0
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            match = _FRAME_RE.match(name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


def _frame_range_text(start, count):
    """'frame 000006' for one image, 'frames 000006-000008' for
    several -- used in status lines and settings headers."""
    if count == 1:
        return "frame {:06d}".format(start)
    return "frames {:06d}-{:06d}".format(start, start + count - 1)


def _seeds_for(seed, count):
    """Resolve the seed box to one seed per image. -1 (or empty) draws
    a fresh random seed per image; a fixed seed S gives S, S+1, S+2...
    so a multi-image run with a fixed seed stays reproducible."""
    if seed is None:
        seed = -1
    seed = int(seed)
    if seed < 0:
        return [random.randint(0, 2**31 - 1) for _ in range(count)]
    return [seed + i for i in range(count)]


# Dynamic prompts: single level only, one random pick per group.
# "{a|b|c}" becomes a, b, or c. Groups do not nest; a group with no
# pipe is just its own text with the braces removed.
_TEMPLATE_RE = re.compile(r"\{([^{}]*)\}")


def _expand_templates(prompt):
    """Expand {a|b|c} groups, one random pick each. Plain prompts
    (no braces) come back unchanged."""
    return _TEMPLATE_RE.sub(
        lambda match: random.choice(match.group(1).split("|")), prompt
    )


def _append_settings(folder, header, entries):
    """Append one run's block to settings.txt: a header line (date,
    time, frame range) followed by the run's 'key: value' lines and
    a blank line. Creates the file on first use; earlier runs' blocks
    are never touched. Old single-run files end with a newline, but
    a missing one is repaired so blocks never run together."""
    path = os.path.join(folder, "settings.txt")
    prefix = ""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                prefix = "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(prefix + header + "\n")
        handle.write("\n".join(entries) + "\n\n")


def _run_header(start, count):
    """The settings.txt header for one run: date, time, and the
    frame-number range it produced."""
    return "Run {}, {}".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        _frame_range_text(start, count))


def _seed_entry(seeds):
    """The settings.txt seed line: 'seed: X' for one image,
    'seeds: X, Y' for several."""
    if len(seeds) == 1:
        return "seed: {}".format(seeds[0])
    return "seeds: {}".format(", ".join(str(s) for s in seeds))


def _seed_text(seeds):
    """The status-line seed text, same wording as settings.txt."""
    return _seed_entry(seeds)


def _expanded_entries(expanded_list):
    """The settings.txt expanded-prompt line(s): one line for one
    image, a comma-joined list for several."""
    if len(expanded_list) == 1:
        return ["expanded_prompt: {}".format(expanded_list[0])]
    return ["expanded_prompts: {}".format(", ".join(expanded_list))]


def _run_generate(pipe, template, negative, width, height, steps,
                  guidance, seeds, folder, start=1):
    """One text-to-image run per seed. Each image gets its own fresh
    dynamic-prompt expansion and its own generator, exactly as the
    single-image flow does, so each image's seed reproduces it.
    Frames are numbered from start (6-digit), so a repeat run in a
    reused folder continues where the last run left off.
    Returns (frame paths, expanded prompts actually used)."""
    import torch

    frame_paths = []
    expanded_list = []
    for i, actual_seed in enumerate(seeds):
        expanded = _expand_templates(template)
        generator = torch.Generator(device="cuda").manual_seed(actual_seed)
        image = pipe(
            prompt=expanded,
            negative_prompt=negative,
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=generator,
        ).images[0]
        frame_path = os.path.join(folder, "{:06d}.png".format(start + i))
        image.save(frame_path)
        frame_paths.append(frame_path)
        expanded_list.append(expanded)
    return frame_paths, expanded_list


def _generate_entries(template, expanded_list, negative_prompt, width,
                      height, steps, guidance, seeds):
    """settings.txt entries for a text-to-image run, in the usual
    order. The expanded prompt line is what was actually used (the
    template line keeps what was typed)."""
    return (
        ["prompt: {}".format(template)]
        + _expanded_entries(expanded_list)
        + ["negative_prompt: {}".format(negative_prompt or "")]
        + ["width: {}".format(int(width)),
           "height: {}".format(int(height)),
           "inference_steps: {}".format(int(steps)),
           "guidance_scale: {}".format(float(guidance))]
        + [_seed_entry(seeds),
           "model: {}".format(MODEL_ID),
           "dtype: bfloat16",
           "device: cuda"]
    )


def _history_add(history, prompt):
    """Prepend a prompt to the session history: most recent first,
    last 10, duplicates collapsed. Never written to disk."""
    history = history or []
    return [prompt] + [p for p in history if p != prompt][:9]


def generate(prompt, negative_prompt, queue_text, width, height,
             steps, guidance, seed, batch_name, images_per_generate,
             last_folder, history):
    """Text-to-image generation. Returns (gallery image list,
    status text, history dropdown update, last batch folder,
    history list). The gallery shows the saved frames.

    If the prompt queue has any non-empty lines, the queue runs
    INSTEAD of the single prompt box: each line is its own run with
    the current settings, sharing the one batch folder, frames
    numbered continuously across lines, one settings block per line.
    Otherwise the single prompt box runs as before. Repeat runs
    with the same batch name reuse the folder and keep numbering
    upward, so nothing is ever overwritten."""
    import torch

    lines = [line.strip() for line in (queue_text or "").splitlines()
             if line.strip()]
    if not lines and (not prompt or not prompt.strip()):
        return [], "Type a prompt first.", gr_update(), last_folder, \
            history
    if not torch.cuda.is_available():
        return [], (
            "No CUDA GPU found. Sana UI needs an NVIDIA GPU with "
            "a recent driver."
        ), gr_update(), last_folder, history
    try:
        pipe = _load_pipe()
    except Exception as exc:  # noqa: BLE001 -- show it, don't crash the UI
        return [], "Model load failed: {}".format(exc), gr_update(), \
            last_folder, history

    # Images per Generate is clamped to 1-20: the box already limits
    # it, but a typed value can slip past, so the code enforces it.
    count = int(images_per_generate) if images_per_generate else 1
    count = max(1, min(20, count))
    negative = (negative_prompt or "").strip()
    history = history or []

    folder = _batch_folder(_safe_batch_name(batch_name))
    os.makedirs(folder, exist_ok=True)
    if lines:
        all_frames = []
        all_seeds = []
        first_frame = None
        for line in lines:
            seeds = _seeds_for(seed, count)
            start = _next_frame_number(folder)
            if first_frame is None:
                first_frame = start
            frames, expanded = _run_generate(
                pipe, line, negative, width, height, steps, guidance,
                seeds, folder, start)
            _append_settings(folder, _run_header(start, len(frames)),
                             _generate_entries(
                                 line, expanded, negative_prompt,
                                 width, height, steps, guidance,
                                 seeds))
            all_frames.extend(frames)
            all_seeds.extend(seeds)
            history = _history_add(history, line)
        status = "Queue done: {} prompts, {} images. Saved to {} ({}, {})".format(
            len(lines), len(all_frames), folder,
            _frame_range_text(first_frame, len(all_frames)),
            _seed_text(all_seeds))
        return (all_frames, status, _history_update(history),
                folder, history)

    seeds = _seeds_for(seed, count)
    start = _next_frame_number(folder)
    frames, expanded = _run_generate(
        pipe, prompt, negative, width, height, steps, guidance,
        seeds, folder, start)
    _append_settings(folder, _run_header(start, len(frames)),
                     _generate_entries(
                         prompt, expanded, negative_prompt, width,
                         height, steps, guidance, seeds))
    status = "Saved to {} ({}, {})".format(
        folder, _frame_range_text(start, len(frames)),
        _seed_text(seeds))
    history = _history_add(history, prompt)
    return (frames, status, _history_update(history), folder, history)


def random_prompt():
    """Assemble one prompt from the in-code word banks. No list file,
    nothing to maintain -- just a starting point to edit or reroll."""
    return "A {} {}, {}, {}".format(
        random.choice(_RANDOM_SUBJECTS),
        random.choice(_RANDOM_PLACES),
        random.choice(_RANDOM_STYLES),
        random.choice(_RANDOM_LIGHTING),
    )


def star_batch(last_folder):
    """Copy the most recently generated batch folder into
    images_out/favorites/ under a unique name (never overwrite).
    If nothing has run yet this session, say so in the status."""
    if not last_folder or not os.path.isdir(last_folder):
        return "No batch has run yet in this session."
    os.makedirs(FAVORITES_DIR, exist_ok=True)
    base = os.path.basename(os.path.normpath(last_folder))
    dest = _unique_folder(base, root=FAVORITES_DIR)
    shutil.copytree(last_folder, dest)
    return "Starred to {}".format(dest)


def _history_update(choices):
    """Dropdown update helper, kept in one place so generate() stays
    readable. Gradio import is deferred to UI build time."""
    import gradio as gr
    return gr.update(choices=choices)


def gr_update():
    """No-change update for the history dropdown (early exits)."""
    import gradio as gr
    return gr.update()


def build_ui():
    import gradio as gr

    with gr.Blocks(title="Sana 1.6B") as demo:
        gr.Markdown(
            "# Sana 1.6B\n"
            "Text-to-image on your own GPU -- no Jupyter, no Colab. "
            "First Generate downloads the model (about 10 GB), so "
            "it takes a while; after that, repeat runs are just the render."
        )
        # Session-only state: never written to disk.
        last_folder_state = gr.State(None)
        history_state = gr.State([])

        with gr.Tab("Text to image"):
            prompt = gr.Textbox(label="Prompt", lines=3,
                                value=DEFAULT_PROMPT)
            negative_prompt = gr.Textbox(label="Negative prompt",
                                         lines=2, value=DEFAULT_NEGATIVE)
            queue_text = gr.Textbox(
                label="Prompt queue (one prompt per line)", lines=4)
            gr.Markdown(
                "When the queue has lines, Generate runs the queue "
                "instead of the prompt above -- one run per line, "
                "all sharing the batch folder, frames numbered "
                "continuously, one settings block per line."
            )
            with gr.Row():
                random_btn = gr.Button("Random prompt")
            with gr.Row():
                width = gr.Slider(512, 1024, step=64, value=1024,
                                  label="Width")
                height = gr.Slider(512, 1024, step=64, value=1024,
                                   label="Height")
            with gr.Row():
                square_btn = gr.Button("Square")
                portrait_btn = gr.Button("Portrait")
                landscape_btn = gr.Button("Landscape")
            with gr.Row():
                steps = gr.Slider(5, 50, step=1, value=20,
                                  label="Inference steps")
                guidance = gr.Slider(1.0, 10.0, step=0.5, value=4.5,
                                     label="Guidance scale")
                seed = gr.Number(value=-1, precision=0,
                                 label="Seed (-1 = random)")
            batch_name = gr.Textbox(label="Batch name",
                                    value="sana-test")
            images_per_generate = gr.Number(
                value=1, precision=0, minimum=1, maximum=20,
                label="Images per Generate")
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary")
                stop_btn = gr.Button("Stop")
                star_btn = gr.Button("Star batch")
            gallery = gr.Gallery(label="Results")
            status = gr.Textbox(label="Status")
            history = gr.Dropdown(label="Prompt history", choices=[],
                                  interactive=True)

        random_btn.click(random_prompt, outputs=[prompt])
        square_btn.click(lambda: (1024, 1024), outputs=[width, height])
        portrait_btn.click(lambda: (768, 1024),
                           outputs=[width, height])
        landscape_btn.click(lambda: (1024, 768),
                            outputs=[width, height])
        generate_event = generate_btn.click(
            generate,
            inputs=[prompt, negative_prompt, queue_text, width,
                    height, steps, guidance, seed, batch_name,
                    images_per_generate, last_folder_state,
                    history_state],
            outputs=[gallery, status, history, last_folder_state,
                     history_state],
        )
        # Gradio 5 cancel pattern: Stop cancels the running Generate
        # event. Queued runs stop at once; a run already diffusing
        # is interrupted by the queue (verify on the PC).
        # Cancelling an event that is not running is a no-op.
        stop_btn.click(fn=None, inputs=[], outputs=[],
                       cancels=[generate_event])
        star_btn.click(star_batch, inputs=[last_folder_state],
                       outputs=[status])
        history.change(fn=lambda choice: choice or "",
                       inputs=[history], outputs=[prompt])
    return demo


if __name__ == "__main__":
    os.makedirs(IMAGES_DIR, exist_ok=True)
    build_ui().launch(server_name="127.0.0.1", inbrowser=True,
                      allowed_paths=[IMAGES_DIR])
