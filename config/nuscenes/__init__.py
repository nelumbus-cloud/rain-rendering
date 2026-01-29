import json
import os

from config.nuscenes.nusc_dataset import NuScenesDataset, NuScenesGANDataset

# Module-level state (initialized in _init)
nusc_dataset = None
root = None


def _sequences(results):
    """
    Returns a list of *scene tokens* (strings) for nuScenes.
    If results.sequences is provided, supports:
      - comma-separated numeric indices into the sorted unique scene token list
      - comma-separated scene tokens directly
    """
    global nusc_dataset
    unique_sequences = sorted(list(set(nusc_dataset.scene_tokens)))

    if results.sequences:
        parts = [p.strip() for p in results.sequences.split(",") if p.strip()]
        is_numeric = parts[0].isnumeric()

        if is_numeric:
            idxs = [int(p) for p in parts if int(p) < len(unique_sequences)]
            sequences = [unique_sequences[i] for i in idxs]
        else:
            # assume tokens
            sequences = parts
    else:
        sequences = unique_sequences

    return sequences


def _init(results):
    """
    Initializes (root, nusc_dataset) based on whether we're using gan or not,
    and optional json_file containing sample_data_tokens.
    """
    global root, nusc_dataset

    tokens = None
    if getattr(results, "json_file", None):
        with open(results.json_file, "r") as f:
            tokens = json.load(f).get("sample_data_tokens", None)

    is_gan = "gan" in results.dataset

    if is_gan:
        root = results.gan_root
        nusc_dataset = NuScenesGANDataset(
            version=results.dataset_version,
            root=results.dataset_root,
            gan_root=results.gan_root,
            post_fix=results.post_fix,
            pretransform_data=False,
            preload_data=False,
            only_annotated=False,
            specific_tokens=tokens,
        )
    else:
        root = results.dataset_root
        nusc_dataset = NuScenesDataset(
            version=results.dataset_version,
            root=results.dataset_root,
            pretransform_data=False,
            preload_data=False,
            only_annotated=False,
            specific_tokens=tokens,
        )


def resolve_paths(results):
    """
    This function adapts nuScenes (token-based) into the repo's expected interface.

    IMPORTANT:
    - results.sequences are scene tokens.
    - results.images[seq] and results.depth[seq] are set to *folder paths* to satisfy
      the legacy checks in main.py (which expects strings, not lists).
      This is a compatibility shim. Proper per-scene file listing should be handled
      in the generator later.
    """
    _init(results)

    results.sequences = _sequences(results)
    assert len(results.sequences) > 0, "There are no valid sequences folder in the dataset root."

    # Folder paths (shared across scenes) – legacy compatibility
    results.images = {
        sequence: os.path.join(root, "samples", "CAM_FRONT")
        for sequence in results.sequences
    }

    results.depth = {
        sequence: os.path.join(results.depth_root, "samples", "CAM_FRONT")
        for sequence in results.sequences
    }

    # Particle simulator preset options (do NOT overwrite results.particles here)
    cameras = ["CAM_FRONT"]
    motions = ["static"]      # may need to match simulator accepted presets
    durations = [1.0]

    results.particles_opts = {
        seq: {"preset": ["nuscenes", seq, cameras, motions, durations]}
        for seq in results.sequences
    }

    results.calib = {sequence: None for sequence in results.sequences}
    return results
     
def settings():
    settings = {}

    # Camera intrinsic-ish parameters used by the simulator (kept from original template)
    settings["cam_focal"] = 5.5          # mm
    settings["cam_gain"] = 1.0
    settings["cam_f_number"] = 1.8
    settings["cam_focus_plane"] = 6.0    # meters
    settings["cam_exposure"] = 5.0       # ms

    # Camera extrinsic parameters
    settings["cam_pos"] = [1.5, 1.5, 0.3]
    settings["cam_lookat"] = [1.5, 1.5, -1.0]
    settings["cam_up"] = [0.0, 1.0, 0.0]

    # Sequence-wise overrides (none for nuScenes in this shim)
    settings["sequences"] = {}
    settings["sequences"][".*"] = {
        "preset": ["nuscenes", "CAM_FRONT", ["CAM_FRONT"], ["static"], [40.0]]
    }

    return settings
