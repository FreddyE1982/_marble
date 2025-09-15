import os
import tempfile
import unittest

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
        print("snapshot image:", result, "size", os.path.getsize(result))
        self.assertTrue(os.path.exists(result))
        self.assertGreater(os.path.getsize(result), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
