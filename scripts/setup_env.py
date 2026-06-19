"""Prefetch every remote artifact used by the tutorial notebooks.

Set ``HF_HUB_OFFLINE=1`` to verify that the cache is complete without making
network requests. GitHub Actions saves the cache immediately after this script
runs, before notebooks can modify extracted working data.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from packaging.version import Version


LAZYSLIDE_DATA_REPO = "RendeiroLab/LazySlide-data"

DATA_FILES: dict[str, tuple[str, ...]] = {
    LAZYSLIDE_DATA_REPO: (
        "sample.svs",
        "sample.zarr.zip",
        "GTEX-1117F-0526.svs",
        "GTEX-1117F-0526.zarr.zip",
        "GTEX-11DXX-1626.svs",
        "GTEX-11DXX-1626.zarr.zip",
        "lung_carcinoma.ndpi",
        "lung_carcinoma.zarr.zip",
        "TCGA_READ_subset_TITAN.h5ad",
        "TCGA_READ_survival.csv",
        "GTEx_artery_RNA.h5ad",
        "agg_conch_features.h5ad",
        "gtex_stomach_subset.zip",
    ),
    "MahmoodLab/hest": (
        "wsis/NCBI776.tif",
        "st/NCBI776.h5ad",
    ),
}

MODEL_FILES: dict[str, tuple[str, ...]] = {
    "RendeiroLab/LazySlide-models-gpl": (
        "PathProfiler/PathProfiler_tissue_seg_exported.pt2",
        "PathProfiler/PathProfiler_patch_quality_exported.pt2",
        "CTransPath/CTransPath_exported.pt2",
    ),
    "RendeiroLab/LazySlide-models": (
        "instanseg/instanseg_v0_1_0.pt",
        "Path2Space/Path2Space_exported.pt2",
        "Path2Space/Path2Space_genes.txt",
    ),
    "Owkin-Bioptimus/histoplus": ("histoplus_cellvit_segmentor_20x.pt",),
    "prov-gigatime/GigaTIME": ("model.pth",),
}

# These repositories are consumed through Transformers, Diffusers, timm, or
# CONCH and need their configs/tokenizers/custom code alongside model weights.
MODEL_REPOSITORIES: tuple[str, ...] = (
    "paige-ai/Prism",
    "vinid/plip",
    "Owkin-Bioptimus/CytoSyn",
    "MahmoodLab/conch",
    "paige-ai/Virchow",
    "bioptimus/H0-mini",
    "timm/resnet50.a1_in1k",
)

_OFFLINE_VALUES = {"1", "ON", "TRUE", "YES"}


def is_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in _OFFLINE_VALUES


def lazyslide_data_revision() -> str | None:
    """Match the revision-selection behavior used by ``lazyslide.datasets``."""
    import lazyslide

    version = Version(lazyslide.__version__)
    tag = f"v{version.base_version}"
    if version.public != version.base_version or version.local is not None:
        return None
    if is_offline():
        return tag

    refs = HfApi().list_repo_refs(LAZYSLIDE_DATA_REPO, repo_type="dataset")
    return tag if tag in {ref.name for ref in refs.tags} else None


def download_files(
    repo_id: str,
    filenames: Iterable[str],
    *,
    repo_type: str | None = None,
    revision: str | None = None,
) -> None:
    for filename in filenames:
        print(f"Prefetching {repo_id}/{filename}")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            local_files_only=is_offline(),
        )


def setup_data() -> None:
    revision = lazyslide_data_revision()
    for repo_id, filenames in DATA_FILES.items():
        download_files(
            repo_id,
            filenames,
            repo_type="dataset",
            revision=revision if repo_id == LAZYSLIDE_DATA_REPO else None,
        )


def setup_models() -> None:
    for repo_id, filenames in MODEL_FILES.items():
        download_files(repo_id, filenames)

    for repo_id in MODEL_REPOSITORIES:
        print(f"Prefetching model repository {repo_id}")
        snapshot_download(repo_id=repo_id, local_files_only=is_offline())


def main() -> None:
    setup_data()
    setup_models()
    mode = "offline verification" if is_offline() else "prefetch"
    print(f"Environment {mode} complete.")


if __name__ == "__main__":
    main()
