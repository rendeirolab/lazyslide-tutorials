"""Build gtex_stomach_subset_compact.zip for the tile model tutorial.

The tutorial undersamples every class to the smallest one (hemorrhage, 7,700
tiles), so it uses about 38,500 of the 1.5M tiles in gtex_stomach_subset.zip
across its train, validation and test splits. This keeps that many training
tiles, drawn with a fixed seed, and the one test slide the tutorial plots.

Usage: uv run scripts/build_stomach_compact.py <extracted gtex-stomach-subset> <out gtex-stomach-subset>
Then zip from the output folder's parent, so the zip holds one top folder named gtex-stomach-subset:
    zip -qr gtex_stomach_subset_compact.zip gtex-stomach-subset

Built with spatialdata 0.7.3 (Zarr v3 train stores) and NumPy's default_rng(0);
other versions may draw different rows. The published zip has sha256
f4c0480272b86ab6ed2f1d8d632843b850b424b21f9f526f7fcf73379562cc7a.
"""

import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd

PER_CLASS = 7_700
SKIP = {"artifact"}
TEST_SLIDE = "GTEX-1JJE9-2026"


def main(src: Path, out: Path) -> None:
    stores = sorted((src / "train").glob("*.zarr"))
    labels = pd.concat(
        pd.DataFrame({"store": i, "row": np.arange(len(d)), "domain": d.to_numpy()})
        for i, s in enumerate(stores)
        for d in [gpd.read_parquet(s / "shapes" / "tiles" / "shapes.parquet")["domain"]]
    )
    labels = labels[~labels["domain"].isin(SKIP)]
    rng = np.random.default_rng(0)
    keep = pd.concat(
        g.iloc[rng.choice(len(g), size=min(PER_CLASS, len(g)), replace=False)]
        for _, g in labels.groupby("domain")
    )
    (out / "train").mkdir(parents=True)
    for i, rows in keep.groupby("store")["row"]:
        rows = np.sort(rows.to_numpy())
        sdata = sd.read_zarr(stores[i])
        sdata.shapes["tiles"] = sdata.shapes["tiles"].iloc[rows]
        sdata.tables["virchow2_tiles"] = sdata.tables["virchow2_tiles"][rows].copy()
        sdata.write(out / "train" / stores[i].name)
    (out / "test").mkdir()
    for f in (src / "test").glob(f"{TEST_SLIDE}*"):
        (shutil.copytree if f.is_dir() else shutil.copy2)(f, out / "test" / f.name)
    print(keep["domain"].value_counts().to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
