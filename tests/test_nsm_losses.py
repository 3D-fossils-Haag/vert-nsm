"""Tests for NSM/losses.py.

Uses analytic SDFs with known gradients instead of a trained network:
  sphere(x)      = |x| - r       -> |grad| = 1 everywhere (a valid SDF, eikonal loss 0)
  2 * sphere(x)                  -> |grad| = 2 everywhere (eikonal loss (2-1)^2 = 1)
"""
import pytest
import torch

from NSM.losses import combined_sdf_loss, compute_sdf_gradients, eikonal_loss, l1_loss, l2_loss

RADIUS = 0.5


def sphere_sdf(points, radius=RADIUS):
    return points.norm(dim=-1) - radius


class SphereModel(torch.nn.Module):
    """Mimics the NSM decoder interface: input is cat([latent, xyz]), output (B, n_surfaces).
    Ignores the latent; surface k is (k + 1) * sphere."""

    def __init__(self, n_surfaces=1):
        super().__init__()
        self.n_surfaces = n_surfaces

    def forward(self, x):
        sdf = sphere_sdf(x[:, -3:])
        return torch.stack([(k + 1) * sdf for k in range(self.n_surfaces)], dim=-1)


@pytest.fixture
def points():
    torch.manual_seed(0)
    # keep away from the origin, where |x| is not differentiable
    pts = torch.randn(256, 3)
    return pts / pts.norm(dim=-1, keepdim=True) * torch.empty(256, 1).uniform_(0.2, 1.5)


# --- eikonal_loss ----------------------------------------------------------------

def test_eikonal_is_zero_for_true_sdf(points):
    pts = points.clone().requires_grad_(True)
    assert eikonal_loss(sphere_sdf(pts), pts).item() == pytest.approx(0.0, abs=1e-10)


def test_eikonal_penalises_scaled_sdf(points):
    pts = points.clone().requires_grad_(True)
    assert eikonal_loss(2 * sphere_sdf(pts), pts).item() == pytest.approx(1.0, rel=1e-5)


def test_eikonal_reductions(points):
    pts = points.clone().requires_grad_(True)
    per_point = eikonal_loss(3 * sphere_sdf(pts), pts, reduction="none")
    assert per_point.shape == (len(points),)
    torch.testing.assert_close(per_point, torch.full_like(per_point, 4.0))

    pts = points.clone().requires_grad_(True)
    total = eikonal_loss(3 * sphere_sdf(pts), pts, reduction="sum")
    assert total.item() == pytest.approx(4.0 * len(points), rel=1e-5)


def test_eikonal_averages_over_surfaces(points):
    pts = points.clone().requires_grad_(True)
    sdf = torch.stack([sphere_sdf(pts), 2 * sphere_sdf(pts)], dim=-1)  # losses 0 and 1
    assert eikonal_loss(sdf, pts).item() == pytest.approx(0.5, rel=1e-5)


def test_eikonal_is_differentiable(points):
    scale = torch.tensor(2.0, requires_grad=True)
    pts = points.clone().requires_grad_(True)
    loss = eikonal_loss(scale * sphere_sdf(pts), pts)
    loss.backward()
    # d/ds (s - 1)^2 = 2(s - 1) = 2 at s = 2
    assert scale.grad.item() == pytest.approx(2.0, rel=1e-5)


def test_eikonal_requires_grad_on_points(points):
    with pytest.raises(ValueError, match="require gradients"):
        eikonal_loss(sphere_sdf(points), points)


def test_eikonal_rejects_unknown_reduction(points):
    pts = points.clone().requires_grad_(True)
    with pytest.raises(ValueError, match="Unknown reduction"):
        eikonal_loss(sphere_sdf(pts), pts, reduction="median")


# --- compute_sdf_gradients ---------------------------------------------------------

def test_gradients_of_sphere_point_radially_outward(points):
    latent = torch.zeros(4)  # 1-D latent is broadcast to every point
    sdf, grads = compute_sdf_gradients(SphereModel(), latent, points, surface_idx=0)
    torch.testing.assert_close(sdf, sphere_sdf(points))
    torch.testing.assert_close(grads, points / points.norm(dim=-1, keepdim=True))


def test_gradients_for_all_surfaces(points):
    sdf, grads = compute_sdf_gradients(SphereModel(n_surfaces=2), torch.zeros(1, 4), points)
    assert sdf.shape == (len(points), 2)
    assert len(grads) == 2
    torch.testing.assert_close(grads[1], 2 * grads[0])


def test_gradients_do_not_modify_input_points(points):
    before = points.clone()
    compute_sdf_gradients(SphereModel(), torch.zeros(4), points, surface_idx=0)
    assert not points.requires_grad
    torch.testing.assert_close(points, before)


# --- combined_sdf_loss -----------------------------------------------------------

def test_combined_reconstruction_only():
    pred, gt = torch.tensor([0.0, 1.0, -1.0]), torch.tensor([0.5, 1.0, 1.0])
    total, parts = combined_sdf_loss(pred, gt, points=None, eikonal_weight=0, loss_type="l1")
    assert total.item() == pytest.approx((0.5 + 0 + 2) / 3)
    assert set(parts) == {"reconstruction"}

    total, _ = combined_sdf_loss(pred, gt, points=None, eikonal_weight=0, loss_type="l2")
    assert total.item() == pytest.approx((0.25 + 0 + 4) / 3)


def test_combined_with_eikonal(points):
    pred = gt = torch.zeros(len(points))
    model = SphereModel(n_surfaces=1)
    total, parts = combined_sdf_loss(pred, gt, points, model=lambda x: 2 * model(x),
                                     latent=torch.zeros(4), eikonal_weight=0.1)
    assert parts["reconstruction"] == pytest.approx(0.0)
    assert parts["eikonal"] == pytest.approx(1.0, rel=1e-5)
    assert total.item() == pytest.approx(0.1, rel=1e-5)


def test_combined_eikonal_needs_model_and_latent(points):
    with pytest.raises(ValueError, match="Model and latent"):
        combined_sdf_loss(torch.zeros(3), torch.zeros(3), points[:3], eikonal_weight=0.1)


def test_combined_rejects_unknown_loss_type():
    with pytest.raises(ValueError, match="Unknown loss_type"):
        combined_sdf_loss(torch.zeros(3), torch.zeros(3), None, eikonal_weight=0, loss_type="huber")


# --- legacy aliases --------------------------------------------------------------

def test_legacy_aliases_match_torch():
    a, b = torch.tensor([1.0, -2.0, 3.0]), torch.tensor([0.0, 0.0, 0.0])
    assert l1_loss(a, b).item() == pytest.approx(2.0)
    assert l2_loss(a, b).item() == pytest.approx(14 / 3)
    assert l1_loss(a, b, reduction="sum").item() == pytest.approx(6.0)
