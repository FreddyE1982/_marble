import unittest

import torch

from marble.plugin_encoder import PluginEncoder
import marble.plugins as plugins


class PluginEncoderTest(unittest.TestCase):
    def test_encode_shapes(self) -> None:
        num_plugins = len(plugins.PLUGIN_ID_REGISTRY)
        self.assertGreater(num_plugins, 0)
        encoder = PluginEncoder(num_plugins, embed_dim=4, action_dim=4, ctx_dim=3)
        # Pick first plugin id
        pid = next(iter(plugins.PLUGIN_ID_REGISTRY.values()))
        plugin_ids = torch.tensor([pid], dtype=torch.long)
        ctx_seq = torch.zeros(1, 2, 3)
        past_ids = torch.tensor([[pid, pid]])
        out = encoder(plugin_ids, ctx_seq, past_ids)
        print("encoded_shape", tuple(out.shape))
        expected_dim = 4 + 3 + 4
        self.assertEqual(out.shape, (1, expected_dim))


if __name__ == "__main__":
    unittest.main()
