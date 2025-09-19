import os
import tempfile
import unittest

from PIL import Image

from marble import snapshot_to_image
from marble.marblemain import Brain
from marble.reporter import clear_report_group


class TestSnapshotVisualization(unittest.TestCase):
    def test_snapshot_to_image_creates_file(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        b = Brain(1, size=3, store_snapshots=True, snapshot_path=tmp, snapshot_freq=1)
        b.add_neuron((0,), tensor=[0.0])
        b.add_neuron((1,), tensor=[1.0], connect_to=(0,), direction="uni")
        snap = b.save_snapshot()
        out_png = os.path.join(tmp, "topology.png")
        result = snapshot_to_image(snap, out_png)
        self.assertTrue(os.path.exists(result))
        with Image.open(result) as img:
            pixels = list(img.convert("L").getdata())
        self.assertGreater(sum(1 for p in pixels if p != 255), 0)

    def test_snapshot_to_image_handles_sparse_snapshot(self):
        clear_report_group("brain")
        tmp = tempfile.mkdtemp()
        b = Brain(
            2,
            mode="sparse",
            sparse_bounds=[(0.0, 1.0), (0.0, 1.0)],
            store_snapshots=True,
            snapshot_path=tmp,
            snapshot_freq=1,
        )
        b.add_neuron((0.1, 0.2), tensor=[0.0])
        b.add_neuron((0.9, 0.8), tensor=[1.0], connect_to=(0.1, 0.2), direction="uni")
        snap = b.save_snapshot()
        out_png = os.path.join(tmp, "sparse_topology.png")
        result = snapshot_to_image(snap, out_png)
        self.assertTrue(os.path.exists(result))
        with Image.open(result) as img:
            pixels = list(img.convert("L").getdata())
        self.assertGreater(sum(1 for p in pixels if p != 255), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
