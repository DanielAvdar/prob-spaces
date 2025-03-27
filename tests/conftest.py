import pytest
import torch

params = [
    "cpu",
]
if torch.cuda.is_available():
    params.append("cuda")


@pytest.fixture(
    params=params,
)
def device(request):
    return request.param
