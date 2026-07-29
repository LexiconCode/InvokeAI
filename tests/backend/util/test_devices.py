"""
Test abstract device class.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from invokeai.app.services.config import get_config
from invokeai.backend.util.devices import TorchDevice, choose_precision, choose_torch_device, torch_dtype

devices = ["cpu", "cuda:0", "cuda:1", "cuda:2", "mps"]
device_types_cpu = [("cpu", torch.float32), ("cuda:0", torch.float32), ("mps", torch.float32)]
device_types_cuda = [("cpu", torch.float32), ("cuda:0", torch.float16), ("mps", torch.float32)]
device_types_mps = [("cpu", torch.float32), ("cuda:0", torch.float32), ("mps", torch.float16)]


@pytest.mark.parametrize("device_name", devices)
def test_device_choice(device_name):
    config = get_config()
    config.device = device_name
    torch_device = TorchDevice.choose_torch_device()
    assert torch_device == torch.device(device_name)


@pytest.mark.parametrize("device_dtype_pair", device_types_cpu)
def test_device_dtype_cpu(device_dtype_pair):
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        device_name, dtype = device_dtype_pair
        config = get_config()
        config.device = device_name
        torch_dtype = TorchDevice.choose_torch_dtype()
        assert torch_dtype == dtype


@pytest.mark.parametrize("device_dtype_pair", device_types_cuda)
def test_device_dtype_cuda(device_dtype_pair):
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_name", return_value="RTX4070"),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        device_name, dtype = device_dtype_pair
        config = get_config()
        config.device = device_name
        torch_dtype = TorchDevice.choose_torch_dtype()
        assert torch_dtype == dtype


@pytest.mark.parametrize("device_dtype_pair", device_types_mps)
def test_device_dtype_mps(device_dtype_pair):
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        device_name, dtype = device_dtype_pair
        config = get_config()
        config.device = device_name
        torch_dtype = TorchDevice.choose_torch_dtype()
        assert torch_dtype == dtype


@pytest.mark.parametrize("device_dtype_pair", device_types_cuda)
def test_device_dtype_override(device_dtype_pair):
    with (
        patch("torch.cuda.get_device_name", return_value="RTX4070"),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        device_name, dtype = device_dtype_pair
        config = get_config()
        config.device = device_name
        config.precision = "float32"
        torch_dtype = TorchDevice.choose_torch_dtype()
        assert torch_dtype == torch.float32


def test_normalize():
    assert (
        TorchDevice.normalize("cuda") == torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cuda")
    )
    assert (
        TorchDevice.normalize("cuda:0") == torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cuda")
    )
    assert (
        TorchDevice.normalize("cuda:1") == torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cuda")
    )
    assert TorchDevice.normalize("mps") == torch.device("mps")
    assert TorchDevice.normalize("cpu") == torch.device("cpu")


@pytest.mark.parametrize("device_name", devices)
def test_legacy_device_choice(device_name):
    config = get_config()
    config.device = device_name
    with pytest.deprecated_call():
        torch_device = choose_torch_device()
    assert torch_device == torch.device(device_name)


@pytest.mark.parametrize("device_dtype_pair", device_types_cpu)
def test_legacy_device_dtype_cpu(device_dtype_pair):
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.get_device_name", return_value="RTX9090"),
    ):
        device_name, dtype = device_dtype_pair
        config = get_config()
        config.device = device_name
        with pytest.deprecated_call():
            torch_device = choose_torch_device()
            returned_dtype = torch_dtype(torch_device)
        assert returned_dtype == dtype


def test_legacy_precision_name():
    config = get_config()
    config.precision = "auto"
    with (
        pytest.deprecated_call(),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.cuda.get_device_name", return_value="RTX9090"),
    ):
        assert "float16" == choose_precision(torch.device("cuda"))
        assert "float16" == choose_precision(torch.device("mps"))
        assert "float32" == choose_precision(torch.device("cpu"))


# ===== choose_anima_inference_dtype (config.precision honoring) ============


def test_choose_anima_inference_dtype_float16():
    """precision='float16' returns torch.float16 without touching hardware."""
    config = get_config()
    config.precision = "float16"
    result = TorchDevice.choose_anima_inference_dtype(torch.device("cpu"))
    assert result is torch.float16


def test_choose_anima_inference_dtype_bfloat16():
    """precision='bfloat16' returns torch.bfloat16 without touching hardware."""
    config = get_config()
    config.precision = "bfloat16"
    result = TorchDevice.choose_anima_inference_dtype(torch.device("cpu"))
    assert result is torch.bfloat16


def test_choose_anima_inference_dtype_float32():
    """precision='float32' returns torch.float32 without touching hardware."""
    config = get_config()
    config.precision = "float32"
    result = TorchDevice.choose_anima_inference_dtype(torch.device("cpu"))
    assert result is torch.float32


def test_choose_anima_inference_dtype_auto_delegates_to_safe_dtype():
    """precision='auto' delegates to choose_bfloat16_safe_dtype (current behavior)."""
    config = get_config()
    config.precision = "auto"
    device = torch.device("cpu")
    sentinel = torch.bfloat16
    with patch.object(TorchDevice, "choose_bfloat16_safe_dtype", return_value=sentinel) as mock_safe:
        result = TorchDevice.choose_anima_inference_dtype(device)
    assert result is sentinel
    mock_safe.assert_called_once_with(device)


# ===== XPU (Intel GPU) ======================================================

device_types_xpu = [
    ("cpu", torch.float32),
    ("cuda:0", torch.float32),
    ("mps", torch.float32),
    ("xpu", torch.float16),
    ("xpu:0", torch.float16),
]


@pytest.mark.parametrize("device_name", ["xpu:0", "xpu:1"])
def test_device_choice_xpu(device_name):
    """An explicit xpu:N device in the config is honored verbatim."""
    config = get_config()
    config.device = device_name
    assert TorchDevice.choose_torch_device() == torch.device(device_name)


def test_auto_device_prefers_xpu_over_cpu():
    """With no CUDA/MPS and an XPU present, `auto` selects (and normalizes) xpu."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
        patch("invokeai.backend.util.devices._xpu_is_available", return_value=True),
        patch("torch.xpu.current_device", return_value=0, create=True),
    ):
        config = get_config()
        config.device = "auto"
        assert TorchDevice.choose_torch_device() == torch.device("xpu", 0)


def test_auto_device_prefers_cuda_over_xpu():
    """CUDA outranks XPU in auto selection."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.current_device", return_value=0),
        patch("invokeai.backend.util.devices._xpu_is_available", return_value=True),
    ):
        config = get_config()
        config.device = "auto"
        assert TorchDevice.choose_torch_device().type == "cuda"


@pytest.mark.parametrize("device_dtype_pair", device_types_xpu)
def test_device_dtype_xpu(device_dtype_pair):
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
        patch("invokeai.backend.util.devices._xpu_is_available", return_value=True),
        patch("torch.xpu.current_device", return_value=0, create=True),
    ):
        device_name, dtype = device_dtype_pair
        config = get_config()
        config.device = device_name
        config.precision = "auto"
        assert TorchDevice.choose_torch_dtype() == dtype


def test_normalize_xpu():
    with (
        patch("invokeai.backend.util.devices._xpu_is_available", return_value=True),
        patch("torch.xpu.current_device", return_value=0, create=True),
    ):
        assert TorchDevice.normalize("xpu") == torch.device("xpu", 0)
        assert TorchDevice.normalize("xpu:1") == torch.device("xpu", 1)
    with patch("invokeai.backend.util.devices._xpu_is_available", return_value=False):
        assert TorchDevice.normalize("xpu") == torch.device("xpu")


# ===== TorchDevice.xpu_mem_get_info fallback ================================


def test_xpu_mem_get_info_native():
    """When torch.xpu.mem_get_info works, its result is passed through."""
    with patch.object(torch.xpu, "mem_get_info", return_value=(5, 10), create=True):
        assert TorchDevice.xpu_mem_get_info(torch.device("xpu")) == (5, 10)


def test_xpu_mem_get_info_fallback_derives_from_properties():
    """Missing SYCL free-memory aspect: derive free/total from total_memory and memory_reserved."""
    gib = 1 << 30
    with (
        patch.object(torch.xpu, "mem_get_info", side_effect=RuntimeError("aspect missing"), create=True),
        patch.object(
            torch.xpu, "get_device_properties", return_value=SimpleNamespace(total_memory=32 * gib), create=True
        ),
        patch.object(torch.xpu, "memory_reserved", return_value=2 * gib, create=True),
    ):
        free, total = TorchDevice.xpu_mem_get_info(torch.device("xpu"))
    assert total == 32 * gib
    assert free == 30 * gib


def test_xpu_mem_get_info_fallback_unknown_total():
    """If even total_memory is unavailable, report (0, 0) rather than raising."""
    with (
        patch.object(torch.xpu, "mem_get_info", side_effect=RuntimeError(), create=True),
        patch.object(torch.xpu, "get_device_properties", side_effect=RuntimeError(), create=True),
        patch.object(torch.xpu, "memory_reserved", side_effect=RuntimeError(), create=True),
    ):
        assert TorchDevice.xpu_mem_get_info(torch.device("xpu")) == (0, 0)
