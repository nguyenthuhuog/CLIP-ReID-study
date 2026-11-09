import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from utils.checkpoint import (
    extract_model_state_dict,
    load_stage2_checkpoint,
    make_stage2_checkpoint,
    restore_rng_state,
    save_checkpoint,
)


class CheckpointTest(unittest.TestCase):
    def test_version_two_checkpoint_round_trips_training_state(self):
        model = torch.nn.Linear(3, 2)
        center_criterion = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer_center = torch.optim.SGD(
            center_criterion.parameters(), lr=0.1
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        checkpoint = make_stage2_checkpoint(
            4,
            model,
            center_criterion,
            optimizer,
            optimizer_center,
            scheduler,
            scaler,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage2.pth"
            save_checkpoint(checkpoint, path)
            loaded = load_stage2_checkpoint(path)

        self.assertEqual(loaded["format_version"], 2)
        self.assertEqual(loaded["stage"], "stage2")
        self.assertEqual(loaded["epoch"], 4)
        self.assertEqual(
            set(extract_model_state_dict(loaded)),
            set(model.state_dict()),
        )
        self.assertIn("rng_state", loaded)

        expected_random = random.random()
        expected_numpy = np.random.rand()
        expected_torch = torch.rand(1)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        restore_rng_state(loaded["rng_state"])
        self.assertEqual(random.random(), expected_random)
        self.assertEqual(np.random.rand(), expected_numpy)
        self.assertTrue(torch.equal(torch.rand(1), expected_torch))

    def test_model_state_extraction_supports_legacy_checkpoints(self):
        legacy_state = {"weight": torch.ones(1)}

        self.assertIs(extract_model_state_dict(legacy_state), legacy_state)

    def test_resume_rejects_legacy_model_only_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pth"
            torch.save({"weight": torch.ones(1)}, path)

            with self.assertRaisesRegex(ValueError, "format version 2"):
                load_stage2_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
