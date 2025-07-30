# This python file is used to set up the environment for executing notebooks.
# Includes data and model weights for the tutorials.

def setup_data():
    from pathlib import Path
    from huggingface_hub import snapshot_download

    snapshot_download(
        "rendeirolab/lazyslide-data",
        repo_type="dataset",
    )

    snapshot_download(
        "rendeirolab/lazyslide-data",
        repo_type="dataset",
        cache_dir=Path(__file__).parent.parent / "tutorials",
    )


def setup_models():
    from lazyslide.models.multimodal.prism import Prism

    _ = Prism()


if __name__ == "__main__":
    setup_data()
    setup_models()

    print("Environment setup complete.")
