brainchop for macOS
===================

This distribution targets Apple Silicon and requires macOS 13.2 or newer.

The installer places the `brainchop` dispatcher and one executable per model in
/usr/local/bin, which is on the default PATH, so they are ready to use immediately:

  brainchop                unified CLI dispatcher (`brainchop -m <model> ...`, default: mindmap)
  brainchop-mindgrab       skull stripping in any modality
  brainchop-mindmap        18-class cortical and subcortical segmentation
  brainchop-mindsnap       104-class Desikan-Killiany parcellation
  brainchop-mindmap-fast   fast lightweight 18-class segmentation

Each executable is entirely self-contained. Its model weights are compiled into
it and it uses no library outside macOS itself, so there is no folder to keep
together, no models directory to install beside it, and nothing else to
download. You may copy a single executable anywhere -- another machine, a
cluster, a USB stick -- and it will run, provided that machine is Apple Silicon
running macOS 13.2 or newer.

Examples:

  brainchop-mindgrab -o stripped.nii.gz input.nii.gz
  brainchop-mindmap -o labels.nii.gz input.nii.gz
  brainchop-mindmap --pve -o brain.nii.gz input.nii.gz
  brainchop-mindsnap -o parcels.nii.gz input.nii.gz

Run any with --help for its option list. Each advertises only the options
its own model supports: brainchop-mindgrab offers --mask, --border and --crop;
the label models offer --save-conform, and brainchop-mindmap also --pve.
Passing an option a sibling has produces a message explaining why it is
unavailable.

Output is written in the input image's own space by default. For
the label models, --save-conform keeps the conformed 256^3 volume
instead.

The default automatic backend on a compatible Mac is Metal. Supplying "-cpu N"
forces the CPU backend and selects N worker threads; values below one let the
program choose, which means this machine's core count.

To uninstall, delete the executables:

  sudo rm /usr/local/bin/brainchop /usr/local/bin/brainchop-mindgrab \
    /usr/local/bin/brainchop-mindmap-fast /usr/local/bin/brainchop-mindmap \
    /usr/local/bin/brainchop-mindsnap
