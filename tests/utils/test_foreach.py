import pytest
import torch

from muonium.utils.foreach import group_tensors_by_shape, move_tensors_to_device

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_2_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires >= 2 CUDA devices"
)


def test_group_tensors_by_shape_groups_by_shape_and_preserves_indices() -> None:
    a = torch.randn(2, 3)
    b = torch.randn(4, 5)
    c = torch.randn(2, 3)

    groups = group_tensors_by_shape([a, b, c])

    assert list(groups) == [(2, 3), (4, 5)]
    assert groups[(2, 3)] == ([a, c], [0, 2])
    assert groups[(4, 5)] == ([b], [1])


def test_group_tensors_by_shape_coalesces_identical_shapes() -> None:
    tensors = [torch.randn(2, 2) for _ in range(4)]

    groups = group_tensors_by_shape(tensors)

    assert groups == {(2, 2): (tensors, [0, 1, 2, 3])}


def test_group_tensors_by_shape_handles_empty_input() -> None:
    assert group_tensors_by_shape([]) == {}


def test_group_tensors_by_shape_uses_empty_tuple_for_scalars() -> None:
    scalar = torch.tensor(1.0)

    assert group_tensors_by_shape([scalar]) == {(): ([scalar], [0])}


def test_group_tensors_by_shape_ignores_dtype_and_device() -> None:
    fp32 = torch.randn(2, 3, dtype=torch.float32)
    bf16 = torch.randn(2, 3, dtype=torch.bfloat16)

    groups = group_tensors_by_shape([fp32, bf16])

    assert groups == {(2, 3): ([fp32, bf16], [0, 1])}


def test_group_tensors_by_shape_compiling_path_preserves_shape_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensors = [torch.randn(2, 3), torch.randn(4, 5)]
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    groups = group_tensors_by_shape(tensors)

    assert groups == {
        (2, 3): ([tensors[0]], [0]),
        (4, 5): ([tensors[1]], [1]),
    }


def test_group_tensors_by_shape_returns_original_tensor_objects() -> None:
    tensors = [torch.randn(2, 3), torch.randn(2, 3)]

    grouped, _ = group_tensors_by_shape(tensors)[(2, 3)]

    assert grouped[0] is tensors[0]
    assert grouped[1] is tensors[1]


def test_move_tensors_to_device_same_device_type_returns_input_object() -> None:
    tensors = [torch.randn(2, 3), None, torch.randn(1)]

    moved = move_tensors_to_device(tensors, torch.device("cpu"), torch.device("cpu"))

    assert moved is tensors


@requires_cuda
def test_move_tensors_to_device_different_type_moves_tensors_and_preserves_none() -> None:
    tensors = [torch.randn(2, 3), None, torch.randn(1)]

    moved = move_tensors_to_device(tensors, torch.device("cpu"), torch.device("cuda"))

    assert moved is not tensors
    assert moved[0] is not None
    assert moved[0].device.type == "cuda"
    assert moved[1] is None
    assert moved[2] is not None
    assert moved[2].device.type == "cuda"
    torch.testing.assert_close(moved[0].cpu(), tensors[0])
    torch.testing.assert_close(moved[2].cpu(), tensors[2])


@requires_cuda
def test_move_tensors_to_device_round_trip_preserves_values() -> None:
    tensors = [torch.randn(2, 3), None, torch.randn(1)]

    on_cuda = move_tensors_to_device(tensors, torch.device("cpu"), torch.device("cuda"))
    back_on_cpu = move_tensors_to_device(on_cuda, torch.device("cuda"), torch.device("cpu"))

    assert back_on_cpu[1] is None
    torch.testing.assert_close(back_on_cpu[0], tensors[0])
    torch.testing.assert_close(back_on_cpu[2], tensors[2])


@requires_2_gpus
def test_move_tensors_to_device_same_type_different_index_is_noop() -> None:
    tensors = [torch.randn(2, 3, device="cuda:0")]

    moved = move_tensors_to_device(tensors, torch.device("cuda:0"), torch.device("cuda:1"))

    assert moved is tensors
    assert moved[0].device == torch.device("cuda:0")


def test_move_tensors_to_device_empty_input() -> None:
    assert move_tensors_to_device([], torch.device("cpu"), torch.device("cuda")) == []
