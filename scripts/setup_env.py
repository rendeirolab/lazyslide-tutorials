# This python file is used to set up the environment for executing notebooks.
# Includes data and model weights for the tutorials.


def setup_data():
    from pathlib import Path
    from huggingface_hub import snapshot_download, hf_hub_download

    snapshot_download(
        "rendeirolab/lazyslide-data",
        ignore_patterns=["gtex_artery_data/*"],
        repo_type="dataset",
    )

    snapshot_download(
        "rendeirolab/lazyslide-data",
        repo_type="dataset",
        ignore_patterns=["gtex_artery_data/*"],
        cache_dir=Path(__file__).parent.parent / "tutorials",
    )

    hf_hub_download(
        "rendeirolab/lazyslide-data",
        "TCGA_READ_subset_TITAN.h5ad",
        repo_type="dataset",
    )

    hf_hub_download(
        "rendeirolab/lazyslide-data",
        "TCGA_READ_survival.csv",
        repo_type="dataset",
    )

    hf_hub_download(
        "RendeiroLab/LazySlide-models-gpl",
        "PathProfiler/pathprofiler_tissue_seg_jit.pt",
    )

    hf_hub_download(
        "RendeiroLab/LazySlide-models",
        "instanseg/instanseg_v0_1_0.pt",
    )


def setup_models():
    from lazyslide.models.multimodal import Prism, PLIP
    from lazyslide.models.segmentation import PathProfilerTissueSegmentation, Instanseg

    _ = Prism()
    _ = PLIP()
    _ = PathProfilerTissueSegmentation()
    _ = Instanseg()


def setup_sample_data():
    import lazyslide as zs

    zs.datasets.sample()
    zs.datasets.gtex_artery()
    zs.datasets.lung_carcinoma()
    zs.datasets.gtex_small_intestine()


if __name__ == "__main__":
    setup_data()
    setup_models()
    setup_sample_data()

    print("Environment setup complete.")
