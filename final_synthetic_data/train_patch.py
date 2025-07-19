from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode
from anomalib.models import Patchcore
from anomalib.engine import Engine
import torch.multiprocessing as mp

# ─── define datamodule & model ─────────────────────────────────────────
datamodule = Folder(
    name="black_crops",
    root="datasets",
    normal_dir="black_crops",
    test_split_mode=TestSplitMode.SYNTHETIC,
    val_split_ratio=0.2,
    train_batch_size=4,
    num_workers=8
)

model  = Patchcore()
engine = Engine()

# ─── main guard required for Windows multiprocessing ───────────────────
if __name__ == "__main__":
    mp.freeze_support()          # optional but nice for PyInstaller users
    mp.set_start_method("spawn", force=True)

    engine.fit(model, datamodule=datamodule)
