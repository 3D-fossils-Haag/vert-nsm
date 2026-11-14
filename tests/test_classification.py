"""Tests for classification_slicer.py, the worker script the Slicer Classification
module launches as a subprocess. It doesn't import `slicer`, so it runs anywhere.

The model, latent optimisation and SDF sampling are mocked: these tests check the
ranking, file outputs and control flow, not the network.
"""
import argparse
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

import classification_slicer

# Reference latents chosen so cosine distance to [1, 0] increases with index.
LATENT_CODES = torch.tensor([
    [0.99, 0.01], [0.90, 0.10], [0.70, 0.30],
    [0.20, 0.80], [0.00, 1.00], [-1.00, 0.00],
])
QUERY_LATENT = torch.tensor([[1.0, 0.0]])
CONFIG = {
    "latent_size": 2,
    "n_pts_per_object": 10,
    "list_mesh_paths": ["/references/mesh_{}.vtk".format(i) for i in range(6)],
}


class _ClassificationTestBase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Patch the names where classification_slicer looks them up (it uses
        # `from NSM... import ...`, so patching NSM.* directly would have no effect).
        patches = {
            "load_config": patch.object(classification_slicer, "load_config", return_value=dict(CONFIG)),
            "load_model": patch.object(classification_slicer, "load_model_and_latents",
                                       return_value=(MagicMock(), None, LATENT_CODES)),
            "top_k": patch.object(classification_slicer, "get_top_k_pcs", return_value=(None, 2)),
            "convert": patch.object(classification_slicer, "convert_ply_to_vtk",
                                    side_effect=lambda path: (None, path)),
            "sdf": patch.object(classification_slicer, "build_sdf_dataset",
                                return_value=(torch.zeros(10, 3), torch.zeros(10, 1), None, None)),
            "optimize": patch.object(classification_slicer, "optimize_latent", return_value=QUERY_LATENT),
            "sdf_patch": patch.object(classification_slicer, "patch_signed_distance_dtype"),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        self.addCleanup(patch.stopall)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _args(self, **overrides):
        args = dict(
            config="config.json", model="model.pth", latent_codes="latents.pth",
            input_mesh=os.path.join(self.temp_dir, "input.vtk"), input_dir=None, fossil_latent=None,
            output_dir=os.path.join(self.temp_dir, "out"),
            result=os.path.join(self.temp_dir, "top5.json"),
            iterations=1, learning_rate=1e-3,
        )
        args.update(overrides)
        return argparse.Namespace(**args)


class TestSingleMesh(_ClassificationTestBase):
    def test_writes_ranked_top_five_json(self):
        args = self._args()
        classification_slicer.classify(args)

        with open(args.result, encoding="utf-8") as stream:
            result = json.load(stream)
        matches = result["matches"]
        self.assertEqual(result["metric"], "cosine distance")
        self.assertEqual([m["rank"] for m in matches], [1, 2, 3, 4, 5])
        self.assertEqual([m["latent_index"] for m in matches], [0, 1, 2, 3, 4])
        self.assertEqual([m["mesh_name"] for m in matches],
                         ["mesh_{}.vtk".format(i) for i in range(5)])
        distances = [m["cosine_distance"] for m in matches]
        self.assertEqual(distances, sorted(distances))
        self.assertAlmostEqual(distances[0], 1 - 0.99 / np.hypot(0.99, 0.01), places=5)
        self.mocks["optimize"].assert_called_once()

    def test_saves_latent_and_indices(self):
        args = self._args()
        classification_slicer.classify(args)

        np.testing.assert_allclose(np.load(os.path.join(args.output_dir, "fossil_latent.npy")),
                                   QUERY_LATENT.numpy())
        np.testing.assert_array_equal(np.load(os.path.join(args.output_dir, "top5_indices.npy")),
                                      [0, 1, 2, 3, 4])
        np.testing.assert_allclose(np.load(os.path.join(args.output_dir, "all_latents.npy")),
                                   LATENT_CODES.numpy())

    def test_precomputed_latent_skips_optimization(self):
        latent_path = os.path.join(self.temp_dir, "precomputed.npy")
        np.save(latent_path, np.array([0.0, 1.0], dtype=np.float32))  # 1-D is accepted
        args = self._args(fossil_latent=latent_path)
        classification_slicer.classify(args)

        self.mocks["optimize"].assert_not_called()
        self.mocks["sdf"].assert_not_called()
        with open(args.result, encoding="utf-8") as stream:
            matches = json.load(stream)["matches"]
        self.assertEqual(matches[0]["latent_index"], 4)  # [0, 1] is closest to [0, 1]

    def test_missing_mesh_names_fall_back_to_unknown(self):
        self.mocks["load_config"].return_value = dict(CONFIG, list_mesh_paths=[])
        args = self._args()
        classification_slicer.classify(args)

        with open(args.result, encoding="utf-8") as stream:
            matches = json.load(stream)["matches"]
        self.assertTrue(all(m["mesh_name"] == "Unknown" for m in matches))

    def test_fewer_than_five_references(self):
        self.mocks["load_model"].return_value = (MagicMock(), None, LATENT_CODES[:3])
        args = self._args()
        classification_slicer.classify(args)

        with open(args.result, encoding="utf-8") as stream:
            self.assertEqual(len(json.load(stream)["matches"]), 3)


class TestBulk(_ClassificationTestBase):
    def _make_input_dir(self, names):
        input_dir = os.path.join(self.temp_dir, "inputs")
        os.makedirs(input_dir)
        for name in names:
            open(os.path.join(input_dir, name), "w").close()
        return input_dir

    def test_list_meshes_filters_and_sorts(self):
        input_dir = self._make_input_dir(["b.vtk", "a.PLY", "notes.txt", "c.vtp", "d.stl"])
        found = [os.path.basename(p) for p in classification_slicer._list_meshes(input_dir)]
        self.assertEqual(found, ["a.PLY", "b.vtk", "c.vtp"])

    def test_bulk_writes_per_mesh_outputs_and_summary(self):
        input_dir = self._make_input_dir(["fossil_a.vtk", "fossil_b.ply"])
        args = self._args(input_mesh=None, input_dir=input_dir)
        classification_slicer.classify(args)

        with open(args.result, encoding="utf-8") as stream:
            summary = json.load(stream)
        self.assertEqual([r["input_name"] for r in summary["results"]], ["fossil_a.vtk", "fossil_b.ply"])
        for base in ("fossil_a", "fossil_b"):
            for suffix in ("_top5.json", "_fossil_latent.npy", "_top5_indices.npy"):
                self.assertTrue(os.path.isfile(os.path.join(args.output_dir, base + suffix)), base + suffix)

    def test_bulk_continues_past_a_failing_mesh(self):
        input_dir = self._make_input_dir(["bad.vtk", "good.vtk"])

        def build(path, *_):
            if os.path.basename(path) == "bad.vtk":
                raise RuntimeError("corrupt mesh")
            return torch.zeros(10, 3), torch.zeros(10, 1), None, None

        self.mocks["sdf"].side_effect = build
        args = self._args(input_mesh=None, input_dir=input_dir)
        classification_slicer.classify(args)

        with open(args.result, encoding="utf-8") as stream:
            summary = json.load(stream)
        self.assertEqual([r["input_name"] for r in summary["results"]], ["good.vtk"])

    def test_bulk_empty_folder_raises(self):
        input_dir = self._make_input_dir(["readme.txt"])
        with self.assertRaises(ValueError):
            classification_slicer.classify(self._args(input_mesh=None, input_dir=input_dir))


if __name__ == "__main__":
    unittest.main()
