"""OSNet architectures (vendored from torchreid, MIT)."""
from . import osnet, osnet_ain

BUILDERS = {
    "osnet_x1_0": osnet.osnet_x1_0,
    "osnet_x0_75": osnet.osnet_x0_75,
    "osnet_x0_5": osnet.osnet_x0_5,
    "osnet_x0_25": osnet.osnet_x0_25,
    "osnet_ibn_x1_0": osnet.osnet_ibn_x1_0,
    "osnet_ain_x1_0": osnet_ain.osnet_ain_x1_0,
    "osnet_ain_x0_75": osnet_ain.osnet_ain_x0_75,
    "osnet_ain_x0_5": osnet_ain.osnet_ain_x0_5,
    "osnet_ain_x0_25": osnet_ain.osnet_ain_x0_25,
}


def build(name: str, num_classes: int = 1000):
    if name not in BUILDERS:
        raise ValueError(f"Unknown OSNet variant {name!r}; choose from {sorted(BUILDERS)}")
    return BUILDERS[name](num_classes=num_classes, pretrained=False, loss="softmax")
