"""Build agg_conch_features.h5ad, the 45-slide table that multiple_slides.ipynb
and genomics_integration.ipynb load.

It runs the steps of multiple_slides.ipynb on every slide in
GTEx_artery_dataset.csv.gz: CONCH features for each tile, their slide mean, and
each term's top-100 score from the tile-to-term similarities. The result has
one row per slide, indexed by Tissue Sample Id in the order of the table: the
slide's mean CONCH feature, its metadata and one score column per term.

Usage: uv run scripts/build_agg_conch_features.py <out.h5ad> [slide ids]
Without slide ids it runs all 45 slides. It writes one zarr store per slide to
data/ in the working directory, as the tutorial does.
"""

import sys

import pandas as pd
import torch
from anndata import read_zarr
from huggingface_hub import hf_hub_download
from wsidata import agg_wsi, open_wsi

import lazyslide as zs

REPO = "RendeiroLab/LazySlide-data"

# The terms of multiple_slides.ipynb.
TERMS = [
    "BMP-2",
    "Monckeberg sclerosis",
    "Runx2",
    "adventitia",
    "apoptosis",
    "arterial hardening",
    "arterial narrowing",
    "arterial remodeling",
    "arterial stiffness",
    "arteriole",
    "artery",
    "atherosclerosis",
    "basement membrane",
    "blood flow",
    "bone morphogenetic protein",
    "calcification",
    "calcified nodule",
    "calcium deposition",
    "calcium phosphate",
    "chronic kidney disease",
    "collagen",
    "compliance",
    "connective tissue",
    "elastic fibers",
    "elasticity",
    "endothelial dysfunction",
    "endothelium",
    "epithelium",
    "external elastic lamina",
    "extracellular matrix",
    "fibroblast",
    "fibrosis",
    "fibrous cap",
    "gap junction",
    "hemodynamics",
    "hydroxyapatite",
    "hyperphosphatemia",
    "inflammation",
    "internal elastic lamina",
    "interstitial space",
    "intima",
    "intimal calcification",
    "intimal thickening",
    "ischemia",
    "lamina propria",
    "lumen",
    "macrocalcification",
    "macrophage",
    "matrix vesicle",
    "mechanotransduction",
    "media",
    "medial calcification",
    "microcalcification",
    "mineralization",
    "myofibroblast",
    "necrotic core",
    "osteoblast-like cell",
    "osteocalcin",
    "osteogenic",
    "osteopontin",
    "oxidative stress",
    "pericyte",
    "phosphate transporter",
    "plaque",
    "shear stress",
    "smooth muscle",
    "tight junction",
    "tunica",
    "vasa vasorum",
    "vascular basement membrane",
    "vascular compliance",
    "vascular integrity",
    "vascular niche",
    "vascular ossification",
    "vascular remodeling",
    "vascular smooth muscle cell",
    "vascular stiffness",
    "vascular tone",
    "vascular wall",
]


def wsi_feature_extraction(slide):
    # The function of the same name in multiple_slides.ipynb.
    s = hf_hub_download(REPO, f"gtex_artery_data/{slide}.svs", repo_type="dataset")
    wsi = open_wsi(s, attach_thumbnail=False, store=f"data/{slide}.zarr")
    zs.pp.find_tissues(wsi)
    zs.pp.tile_tissues(wsi, 256, mpp=0.5, background_fraction=0.5)

    # conch feature
    zs.tl.feature_extraction(wsi, "conch", pbar=False)
    zs.tl.feature_aggregation(wsi, "conch")
    embed = zs.tl.text_embedding(TERMS, "conch")
    zs.tl.text_image_similarity(wsi, embed, "conch")
    wsi.write()


def main(out, slides):
    table = pd.read_csv(
        hf_hub_download(REPO, "GTEx_artery_dataset.csv.gz", repo_type="dataset")
    )
    if slides:
        table = table[table["Tissue Sample Id"].isin(slides)]
    table = table.assign(store=[f"data/{s}.zarr" for s in table["Tissue Sample Id"]])

    slide_scores = {}
    for slide, store in zip(table["Tissue Sample Id"], table["store"]):
        wsi_feature_extraction(slide)
        adata = read_zarr(f"{store}/tables/conch_tiles_text_similarity")
        scores = zs.metrics.topk_score(adata, k=100)
        slide_scores[slide] = dict(zip(adata.var.index, scores))
        print(f"{slide}: {adata.n_obs} tiles", flush=True)

    agg_data = agg_wsi(table, "conch", store_col="store", agg_key="agg_slide")
    agg_data.obs = (
        agg_data.obs.join(pd.DataFrame(slide_scores).T, on="Tissue Sample Id")
        .drop(columns="store")
        .set_index("Tissue Sample Id")
    )
    agg_data.write_h5ad(out)


if __name__ == "__main__":
    # The tutorial renders on CPU. Turn off TF32 convolutions so features
    # computed on a GPU stay close to the CPU ones.
    torch.backends.cudnn.allow_tf32 = False
    main(sys.argv[1], sys.argv[2:])
