from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode
from anomalib.models import EfficientAd
from anomalib.engine import Engine
import torch.multiprocessing as mp

# ─── define datamodule & model ─────────────────────────────────────────
datamodule = Folder(
    name="black_crops",
    root="datasets/black_crops",
    normal_dir="good",
    test_split_mode=TestSplitMode.SYNTHETIC,
    val_split_ratio=0.2,
    train_batch_size=1,
    num_workers=8
)

model  = EfficientAd(imagenet_dir="datasets/imagenette")
engine = Engine(max_epochs=50)

# ─── main guard required for Windows multiprocessing ───────────────────
if __name__ == "__main__":
    mp.freeze_support()          # optional but nice for PyInstaller users
    mp.set_start_method("spawn", force=True)

    engine.fit(model, datamodule=datamodule)
