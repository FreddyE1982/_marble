import os
import shutil
import tempfile
import types
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
from unittest import mock

import marble.hf_utils as hf_utils
import datasets as ds


class HFManualStreamingTest(unittest.TestCase):
    def test_parquet_streaming_and_cleanup(self):
        tmpdir = tempfile.mkdtemp()
        repo_dir = os.path.join(tmpdir, "repo")
        train_dir = os.path.join(repo_dir, "train")
        os.makedirs(train_dir, exist_ok=True)
        pq.write_table(pa.table({"txt": ["a"]}), os.path.join(train_dir, "0000.parquet"))
        pq.write_table(pa.table({"txt": ["b"]}), os.path.join(train_dir, "0001.parquet"))
        downloaded = []

        class HfApi:
            def list_repo_files(self, repo_id, repo_type="dataset"):
                return ["train/0000.parquet", "train/0001.parquet"]

        def hf_hub_download(repo_id, filename, repo_type="dataset", local_dir=None, local_dir_use_symlinks=False):
            src = os.path.join(repo_dir, filename)
            if local_dir is None:
                local_dir = tempfile.mkdtemp(dir=tmpdir)
            os.makedirs(local_dir, exist_ok=True)
            dst = os.path.join(local_dir, os.path.basename(filename))
            shutil.copy(src, dst)
            downloaded.append(dst)
            return dst

        hub = types.SimpleNamespace(HfApi=HfApi, hf_hub_download=hf_hub_download)

        with mock.patch.object(hf_utils, "_ensure_hf_imports", return_value=(hub, ds)):
            wrapper = hf_utils.load_hf_streaming_dataset("dummyrepo", split="train")
            texts = [ex.get_raw("txt") for ex in wrapper]

        self.assertEqual(texts, ["a", "b"])
        for path in downloaded:
            self.assertFalse(os.path.exists(path))
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
