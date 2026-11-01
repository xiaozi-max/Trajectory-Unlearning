import torch

from risk_aware_girpo.simnpo import simnpo_forget_loss


def test_simnpo_is_length_normalized():
    one = simnpo_forget_loss(torch.tensor([[2.0]]), torch.tensor([[1.0]]))
    two = simnpo_forget_loss(torch.tensor([[2.0, 2.0]]), torch.tensor([[1.0, 1.0]]))
    torch.testing.assert_close(one, two)
