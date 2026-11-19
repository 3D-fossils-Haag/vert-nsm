"""Tests for NSM/fossils.py.

Strategy: apply a known similarity transform or a known broken region to a synthetic mesh
and check each function recovers, undoes or masks it. No trained model or real meshes needed.
"""
import nrrd
import numpy as np
import pytest
import pyvista as pv
import torch
from scipy.spatial.transform import Rotation

from NSM.fossils import (
    load_broken_seg,
    mask_broken_region,
    rotation_matrix,
    snap_to_partial,
    undo_pose,
    undo_pose_error,
)


def random_rotation(seed):
    return Rotation.random(random_state=seed).as_matrix()


def asymmetric_mesh():
    """A closed mesh with no rotational symmetry, so rigid fits have a unique answer."""
    mesh = pv.ParametricEllipsoid(0.6, 0.35, 0.2, u_res=40, v_res=40).triangulate().clean()
    mesh.points = mesh.points + 0.1 * np.array([1, 0, 0]) * (mesh.points[:, [1]] ** 2)
    return mesh


# --- rotation_matrix (axis-angle) -------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_rotation_matrix_matches_scipy(seed):
    rotvec = Rotation.random(random_state=seed).as_rotvec()
    R = rotation_matrix(torch.tensor(rotvec, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(R, Rotation.from_rotvec(rotvec).as_matrix(), atol=1e-10)


def test_rotation_matrix_zero_is_identity():
    R = rotation_matrix(torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(R, torch.eye(3, dtype=torch.float64))


def test_rotation_matrix_is_differentiable_at_zero():
    r = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    rotation_matrix(r).sum().backward()
    assert torch.isfinite(r.grad).all()


# --- undo_pose ------------------------------------------------------------------

def test_undo_pose_inverts_forward_pose():
    mesh = asymmetric_mesh()
    pose = torch.tensor([0.1, -0.2, 0.05, 0.02, -0.03, 0.01, np.log(1.1)], dtype=torch.float64)
    R = rotation_matrix(pose[:3]).numpy()
    s = float(torch.exp(pose[6]))
    moved = mesh.copy()
    moved.points = s * mesh.points @ R.T + pose[3:6].numpy()  # forward model in optimize_latent_pose

    restored = undo_pose(moved, pose)
    np.testing.assert_allclose(restored.points, mesh.points, atol=1e-5)
    np.testing.assert_allclose(moved.points, s * mesh.points @ R.T + pose[3:6].numpy())  # input untouched


# --- undo_pose_error ------------------------------------------------------------

def test_undo_pose_error_maps_completion_into_reference_frame(tmp_path):
    ref = asymmetric_mesh()
    R, s, t = random_rotation(7), 1.3, np.array([0.5, 0.1, -0.2])
    fossil = ref.copy()
    fossil.points = s * ref.points @ R + t
    fossil_path, ref_path = str(tmp_path / "fossil.vtk"), str(tmp_path / "ref.vtk")
    fossil.save(fossil_path)
    ref.save(ref_path)

    # a completion that sits exactly on the aligned fossil should land exactly on the reference
    out = undo_pose_error(fossil.copy(), fossil_path, ref_path)
    np.testing.assert_allclose(out.points, ref.points, atol=1e-4)


def test_undo_pose_error_without_reference_is_noop(tmp_path):
    mesh = asymmetric_mesh()
    assert undo_pose_error(mesh, "unused.vtk", None) is mesh
    assert undo_pose_error(mesh, "unused.vtk", str(tmp_path / "missing.vtk")) is mesh


def test_undo_pose_error_with_mismatched_meshes_is_noop(tmp_path):
    fossil_path, ref_path = str(tmp_path / "fossil.vtk"), str(tmp_path / "ref.vtk")
    pv.Sphere(theta_resolution=10).save(fossil_path)
    pv.Sphere(theta_resolution=20).save(ref_path)
    mesh = asymmetric_mesh()
    assert undo_pose_error(mesh, fossil_path, ref_path) is mesh


# --- snap_to_partial --------------------------------------------------------------

def test_snap_to_partial_undoes_small_rigid_offset():
    """snap_to_partial is a local ICP refinement: it should exactly undo small offsets
    (~2 deg, ~1% of bone size). Larger offsets can land in a local minimum, which is
    expected and not tested here."""
    completion = asymmetric_mesh()
    # the "fossil" is one end of the true shape, slightly rotated and shifted
    partial = completion.points[completion.points[:, 0] > 0.1]
    R, t = Rotation.from_euler("xyz", [2, -2, 2], degrees=True).as_matrix(), np.array([0.01, -0.01, 0.01])
    partial = partial @ R + t

    snapped = snap_to_partial(completion, partial)

    np.testing.assert_allclose(snapped.points, completion.points @ R + t, atol=1e-6)
    np.testing.assert_allclose(completion.points, asymmetric_mesh().points)  # input untouched


# --- load_broken_seg ------------------------------------------------------------

SPACING, ORIGIN = 0.5, np.array([1.0, 2.0, 3.0])


def write_seg(path, seg, space="left-posterior-superior", segments=(("other", 1, 0), ("broken", 2, 0))):
    """Labelmap written like Slicer: several segments, each with a label value and layer."""
    header = {"space": space, "space directions": np.eye(3) * SPACING, "space origin": ORIGIN}
    if seg.ndim == 4:
        header["space directions"] = np.vstack([np.full(3, np.nan), np.eye(3) * SPACING])   # no direction for the layer axis
        header["kinds"] = ["list", "domain", "domain", "domain"]
    for i, (name, value, layer) in enumerate(segments):
        header.update({f"Segment{i}_Name": name, f"Segment{i}_LabelValue": str(value), f"Segment{i}_Layer": str(layer)})
    nrrd.write(str(path), seg, header)


def labelmap():
    """10^3 voxels: 'broken' (value 2) where i < 5, 'other' (value 1) where i >= 5."""
    seg = np.ones((10, 10, 10), np.uint8)
    seg[:5] = 2
    return seg


def voxel_centers(ijk):
    return ORIGIN + np.asarray(ijk, float) * SPACING


def test_load_broken_seg_picks_the_broken_segment_by_name(tmp_path):
    write_seg(tmp_path / "a.seg.nrrd", labelmap())
    pts = voxel_centers([[2, 3, 4], [7, 3, 4], [4, 9, 0], [5, 0, 0]])
    np.testing.assert_array_equal(load_broken_seg(str(tmp_path / "a.seg.nrrd"), pts), [1, 0, 1, 0])


def test_load_broken_seg_outside_the_labelmap_is_intact(tmp_path):
    write_seg(tmp_path / "a.seg.nrrd", labelmap())
    pts = voxel_centers([[-3, 0, 0], [2, 20, 0]])
    np.testing.assert_array_equal(load_broken_seg(str(tmp_path / "a.seg.nrrd"), pts), [0, 0])


def test_load_broken_seg_ras_file_with_lps_points(tmp_path):
    # same labelmap saved in RAS: an LPS point (x, y, z) sits at RAS (-x, -y, z)
    write_seg(tmp_path / "ras.seg.nrrd", labelmap(), space="right-anterior-superior")
    ras = voxel_centers([[2, 3, 4], [7, 3, 4]])
    lps = ras * [-1, -1, 1]
    np.testing.assert_array_equal(load_broken_seg(str(tmp_path / "ras.seg.nrrd"), lps), [1, 0])


def test_load_broken_seg_reads_the_broken_layer(tmp_path):
    # overlapping segments in separate layers: 'broken' is value 1 on layer 1
    layers = np.zeros((2, 10, 10, 10), np.uint8)
    layers[0] = 1                         # 'other' covers everything on layer 0
    layers[1, :5] = 1                     # 'broken' on layer 1
    write_seg(tmp_path / "layers.seg.nrrd", layers, segments=(("other", 1, 0), ("broken", 1, 1)))
    pts = voxel_centers([[2, 3, 4], [7, 3, 4]])
    np.testing.assert_array_equal(load_broken_seg(str(tmp_path / "layers.seg.nrrd"), pts), [1, 0])


def test_load_broken_seg_without_named_segment_uses_any_label(tmp_path):
    seg = np.zeros((10, 10, 10), np.uint8)
    seg[:5] = 3
    write_seg(tmp_path / "unnamed.seg.nrrd", seg, segments=())
    pts = voxel_centers([[2, 3, 4], [7, 3, 4]])
    np.testing.assert_array_equal(load_broken_seg(str(tmp_path / "unnamed.seg.nrrd"), pts), [1, 0])


# --- mask_broken_region -----------------------------------------------------------

CENTER, RADIUS = np.array([10.0, -5.0, 2.0]), 3.0


def fossil_with_broken_cap(tmp_path, labels=True):
    """Sphere in world coordinates; vertices with z above the centre + 1 are broken."""
    mesh = pv.Sphere(radius=RADIUS, center=CENTER, theta_resolution=40, phi_resolution=40)
    broken = mesh.points[:, 2] > CENTER[2] + 1.0
    if labels:
        mesh.point_data["broken"] = broken.astype(np.uint8)
    path = str(tmp_path / "fossil.vtk")
    mesh.save(path)
    return path, mesh, broken


def sdf_samples(mesh, n=4000, seed=0):
    """Samples just off the surface, normalized like build_sdf_dataset (world = pts * radius + center)."""
    rng = np.random.default_rng(seed)
    world = mesh.points[rng.integers(0, mesh.n_points, n)] + rng.normal(0, 0.05, (n, 3))
    pts = torch.tensor((world - CENTER) / RADIUS, dtype=torch.float32)
    sdf = torch.tensor(np.arange(n, dtype=np.float32)).reshape(-1, 1)   # sample id, to check pairing
    return pts, sdf, world


def test_mask_drops_samples_nearest_broken_surface(tmp_path):
    path, mesh, broken = fossil_with_broken_cap(tmp_path)
    pts, sdf, world = sdf_samples(mesh)
    kept_pts, kept_sdf = mask_broken_region(pts, sdf, path, CENTER, RADIUS)

    ids = kept_sdf.numpy().ravel().astype(int)
    nearest = np.argmin(((world[:, None] - mesh.points[None]) ** 2).sum(-1), axis=1)
    np.testing.assert_array_equal(ids, np.flatnonzero(~broken[nearest]))         # exactly the intact samples
    np.testing.assert_allclose(kept_pts.numpy(), pts.numpy()[ids])                 # points stay paired with sdfs


def test_mask_subsamples_to_n_samples_reproducibly(tmp_path):
    path, mesh, _ = fossil_with_broken_cap(tmp_path)
    pts, sdf, _ = sdf_samples(mesh)
    a = mask_broken_region(pts, sdf, path, CENTER, RADIUS, n_samples=240)[1]
    b = mask_broken_region(pts, sdf, path, CENTER, RADIUS, n_samples=240)[1]
    assert a.shape == (240, 1)
    np.testing.assert_array_equal(a.numpy(), b.numpy())
    all_intact = mask_broken_region(pts, sdf, path, CENTER, RADIUS)[1].numpy().ravel()
    assert set(a.numpy().ravel()) <= set(all_intact)                               # drawn from intact samples only


def test_mask_without_labels_keeps_everything(tmp_path):
    path, mesh, _ = fossil_with_broken_cap(tmp_path, labels=False)
    pts, sdf, _ = sdf_samples(mesh, n=500)
    kept_pts, kept_sdf = mask_broken_region(pts, sdf, path, CENTER, RADIUS)
    np.testing.assert_array_equal(kept_sdf.numpy(), sdf.numpy())