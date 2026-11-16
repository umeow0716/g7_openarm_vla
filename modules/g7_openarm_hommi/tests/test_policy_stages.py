from __future__ import annotations

import numpy as np
import pytest
import torch
from hommi_diffusion_policy.encoder.dit_obs3d import farthest_point_sample

from g7_openarm_hommi.policy import (
    DEFAULT_ARTIFACT_NAME,
    HommiPolicyRunner,
    farthest_point_indices,
    module_model_dir,
)


def test_farthest_point_indices_match_hommi_sampler():
    points = torch.randn(2, 96, 30, generator=torch.Generator().manual_seed(0))
    points[:, 10:30] = 0.0  # masked patch tokens are all identical
    expected = farthest_point_sample(points, 48, 0)
    assert torch.equal(farthest_point_indices(points, 48), expected)


def test_farthest_point_indices_reject_too_many_samples():
    with pytest.raises(ValueError, match="cannot sample"):
        farthest_point_indices(torch.zeros(1, 4, 3), 5)


def test_staged_inference_matches_hommi_predict_action():
    """The stage pipeline shared by both backends is HoMMI's ``predict_action``."""
    if not (module_model_dir() / DEFAULT_ARTIFACT_NAME).is_file():
        pytest.skip("ego3d artifact not converted")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    runner = HommiPolicyRunner(inference_precision="fp32", backend="torch")
    generator = torch.Generator(device=runner.device).manual_seed(0)
    obs = runner.dummy_obs()
    for key, value in obs.items():
        sample = torch.rand(value.shape, generator=generator, device=runner.device)
        obs[key] = (sample > 0.3).float() if key.endswith("_mask") else sample

    torch.manual_seed(0)
    with torch.inference_mode():
        expected = runner.policy.predict_action(obs)["action_pred"][0].cpu().numpy()
    torch.manual_seed(0)
    np.testing.assert_allclose(runner.predict_horizon(obs), expected, atol=1e-4)
