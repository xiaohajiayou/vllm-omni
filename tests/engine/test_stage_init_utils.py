import pytest
from pytest_mock import MockerFixture

from vllm_omni.engine.stage_init_utils import acquire_device_locks


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_acquire_device_locks_raises_when_visible_devices_are_insufficient(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    mock_platform = mocker.MagicMock()
    mock_platform.device_control_env_var = "CUDA_VISIBLE_DEVICES"
    mock_platform.get_device_count.return_value = 8
    monkeypatch.setattr("vllm_omni.engine.stage_init_utils.current_omni_platform", mock_platform)

    with pytest.raises(RuntimeError, match="requires 4 device\\(s\\).*only 2 device\\(s\\) are available"):
        acquire_device_locks(
            stage_id=0,
            engine_args_dict={
                "parallel_config": {
                    "tensor_parallel_size": 4,
                }
            },
        )
