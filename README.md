# Sana UI

**Sana 1.6B, without Jupyter and without Colab.**

Sana is a fast text-to-image model with a look of its own -- a
different engine from Disco Diffusion and from Kandinsky, not a
replacement for either. This is a plain app: run it, a page opens
in your browser, type a prompt, pictures appear. Built the same
way as Kandinsky UI, free forever.

## What it is

Sana is NVIDIA's text-to-image model. The 1.6B version here uses a
linear diffusion transformer, a Gemma2-2B-IT text encoder, and a
DC-AE decoder that compresses pictures 32 times smaller than usual
-- which is why it is fast and light for its size. It generates
1024px pictures across many sizes and aspect ratios, in English,
Chinese, and emoji prompts. It is not perfect at photorealism,
complex legible text, or hands -- that is the model's own stated
limit, not a bug in this app.

## What it does

- Text-to-image generation on your own GPU -- no cloud, no account
- Prompt + negative prompt, width/height, inference steps,
  guidance scale, seed, batch name
- Up to 20 pictures per Generate, each with its own seed, saved as
  000001.png, 000002.png, and so on
- Square / Portrait / Landscape size presets, a Random prompt button
  (word banks built in, no list file), and a session-only prompt
  history dropdown
- A Stop button to cancel a run, and Star batch to copy a batch into
  images_out/favorites/
- A prompt queue (one prompt per line -- when it has lines, Generate
  runs the queue instead of the single prompt box, all lines sharing
  the one batch folder, frames numbered continuously)
- Dynamic prompts: {a|b|c} groups, one random pick per group, in the
  prompt box and in every queue line. settings.txt records the
  expanded prompt actually used, not the template
- Every run adds its pictures to the batch folder and appends a block
  to `settings.txt` next to them (date, time, frame range, settings)

No animation, no upscaling. Image to image, inpainting, prompt
blending, and ControlNet are NOT in this version -- they are noted
as follow-ups, not built here. (ControlNet exists for Sana in
Diffusers, but it needs a separate ControlNet checkpoint and its
own tab; it did not fit the simple first version.)

## Requirements

- Windows, an NVIDIA GPU (12 GB recommended), and a recent driver
- Python 3.10 or newer, with "Add python.exe to PATH" ticked
- Disk space: about 20 GB free. The app itself is small; the Sana
  model (about 10 GB of bf16 weights) downloads on first Generate
  into the Hugging Face cache, and your pictures live in `images_out/`

No account, no token, no license form: the model repo is public
and not gated. The Gemma text-encoder weights ship inside that
repo, so nothing is fetched from Google.

## Install

1. Extract this folder anywhere you like.
2. Double-click `install.bat` and wait -- it builds a virtual
   environment and installs PyTorch with CUDA plus the packages.
   Run it again if anything interrupts it; it resumes where it
   stopped.
3. Double-click `run.bat`. A page opens in your browser. Type a
   prompt and hit Generate.

## First test

1. Leave everything at the defaults and hit Generate: 1024x1024,
   20 inference steps, guidance 4.5.
2. The first run downloads the model (about 10 GB), so it takes a
   while -- that is the one-time download, not a stuck render. After
   that, repeat runs are just the render.
3. When the 1024 test works, try other sizes and steps. Sana is a
   few-steps model: 20 steps is the standard default, and going far
   past it buys little. If a run ever runs out of memory, close
   other programs using the GPU and try again.

Sana's prompt budget is about 300 tokens -- much bigger than the
77-token limit on CLIP-based models -- so long prompts are welcome.
Past 300 tokens the tail is truncated, same as any model.

## Where pictures go

`images_out/` next to `run.bat`, one subfolder per batch name, reused
across runs -- repeat runs with the same batch name add pictures to
the same folder, numbered continuously (`000001.png`, `000002.png`,
...), so nothing is ever overwritten. `settings.txt` in the folder
grows by one block per run (date, time, frame range, prompt, negative
prompt, size, steps, guidance, seed(s), and model id), so you can
always see exactly what made a picture you like. For dynamic prompts
the expanded prompt actually used is recorded too. Starred batches
land in `images_out/favorites/`.

## How it works

One pipeline, all in bf16 on CUDA, following the model card's own
usage:

1. `SanaPipeline`
   (`Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers`)
   loads with `variant="bf16"`; the VAE and text encoder are moved
   to bf16 after loading.
2. The Gemma2-2B-IT text encoder turns the prompt into embeddings.
3. The Sana transformer denoises a latent in a few steps
   (DPMSolver scheduler).
4. The DC-AE decoder turns the latent into the picture.

Sana generates at the nearest supported ratio internally and
resizes back to the size you asked for, so output dimensions match
the width/height boxes.

## Models and permissions

The models belong to their creators -- see their Hugging Face pages
for terms (Apache 2.0 for Sana, plus the Gemma Terms of Use for the
Gemma text encoder). Nothing is bundled here and nothing is sold.
This UI (the Gradio app, the wrapper, the installer) is by Alan
Carmack.

## License

MIT for this UI code -- free to use, share, and change.
