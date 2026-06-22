import socket
import time

import pytest
import torch
from testkit import _as_tensors, _print_table, assert_close, clone_args, run_example
from torch.multiprocessing.spawn import ProcessRaisedException

from test_support.distributed import _find_free_port, run_distributed

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _noop_worker(rank: int, world_size: int) -> None:
    assert 0 <= rank < world_size


def _assertion_failure_worker(rank: int, world_size: int) -> None:
    assert False, f"rank {rank} failed intentionally"


def _args_kwargs_worker(
    rank: int, world_size: int, expected_arg: str, *, expected_kwarg: int
) -> None:
    assert expected_arg == "forwarded"
    assert expected_kwarg == 7


def _env_worker(rank: int, world_size: int) -> None:
    import os

    assert os.environ["RANK"] == str(rank)
    assert os.environ["LOCAL_RANK"] == str(rank)
    assert os.environ["WORLD_SIZE"] == str(world_size)
    assert os.environ["MASTER_ADDR"] == "127.0.0.1"
    assert int(os.environ["MASTER_PORT"]) > 0


def _uneven_worker(rank: int, world_size: int) -> None:
    if rank == 0:
        time.sleep(0.1)


def test_clone_args_clones_tensors_and_preserves_requires_grad() -> None:
    original = torch.randn(2, 3, requires_grad=True)
    marker = object()

    cloned_tensor, cloned_marker = clone_args((original, marker))

    assert isinstance(cloned_tensor, torch.Tensor)
    assert cloned_tensor is not original
    assert cloned_tensor.requires_grad
    assert cloned_marker is marker
    with torch.no_grad():
        cloned_tensor.add_(1)
    assert not torch.equal(cloned_tensor, original)


def test_clone_args_returns_fresh_tuple_and_tensors_each_call() -> None:
    original = torch.randn(2, 3)

    first = clone_args((original,))
    second = clone_args((original,))

    assert first is not second
    assert first[0] is not second[0]


def test_assert_close_uses_float32_for_low_precision_inputs() -> None:
    actual = torch.tensor([1.0, 1.01], dtype=torch.bfloat16)
    expected = torch.tensor([1.0, 1.0], dtype=torch.bfloat16)

    assert_close(actual, expected, rtol=0.02, atol=0.0)


def test_assert_close_strict_mode_raises_on_any_mismatch() -> None:
    with pytest.raises(AssertionError):
        assert_close(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]), rtol=0, atol=0)


def test_assert_close_mismatch_percentage_boundary_and_message() -> None:
    actual = torch.tensor([0.0, 0.0, 10.0, 10.0])
    expected = torch.zeros(4)

    assert_close(actual, expected, rtol=0, atol=0, max_mismatch_pct=50.0)
    with pytest.raises(AssertionError, match="(?s)custom.*75.0000%"):
        assert_close(
            torch.tensor([0.0, 10.0, 10.0, 10.0]),
            expected,
            rtol=0,
            atol=0,
            max_mismatch_pct=50.0,
            msg="custom",
        )


def test_assert_close_handles_empty_tensors_with_mismatch_budget() -> None:
    assert_close(torch.empty(0), torch.empty(0), rtol=0, atol=0, max_mismatch_pct=0.0)


def test_as_tensors_normalizes_tensor_sequences_and_rejects_non_tensors() -> None:
    x = torch.randn(1)
    y = torch.randn(1)

    assert _as_tensors(x) == [x]
    assert _as_tensors((x, y)) == [x, y]
    assert _as_tensors([x, y]) == [x, y]
    with pytest.raises(AssertionError, match="expected Tensor"):
        _as_tensors((x, "not a tensor"))


def test_run_example_matching_kernel_passes_and_skips_benchmark_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.randn(4)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = run_example(lambda t: t + 1, lambda t: t + 1, (x,))

    assert result == {}


def test_run_example_diverging_kernel_raises_with_label() -> None:
    x = torch.randn(4)

    with pytest.raises(AssertionError, match="kernel: forward output 0"):
        run_example(lambda t: t + 1, lambda t: t, (x,), benchmark=False, rtol=0, atol=0)


def test_run_example_output_count_mismatch_raises() -> None:
    x = torch.randn(4)

    with pytest.raises(AssertionError, match="returned 2 tensors"):
        run_example(lambda t: (t, t), lambda t: t, (x,), benchmark=False)


def test_run_example_first_baseline_is_reference_for_extra_baselines() -> None:
    x = torch.randn(4)

    with pytest.raises(AssertionError, match="shifted_ref: forward output 0"):
        run_example(
            {"kernel": lambda t: t},
            {"ref": lambda t: t, "shifted_ref": lambda t: t + 1},
            (x,),
            benchmark=False,
            rtol=0,
            atol=0,
        )


def test_run_example_checks_multiple_kernels() -> None:
    x = torch.randn(4)

    with pytest.raises(AssertionError, match="bad: forward output 0"):
        run_example(
            {"good": lambda t: t, "bad": lambda t: t + 1},
            lambda t: t,
            (x,),
            benchmark=False,
            rtol=0,
            atol=0,
        )


def test_run_example_clones_inputs_for_each_function() -> None:
    x = torch.zeros(4)

    def mutating_kernel(t: torch.Tensor) -> torch.Tensor:
        t.add_(1)
        return t

    with pytest.raises(AssertionError):
        run_example(mutating_kernel, lambda t: t, (x,), benchmark=False, rtol=0, atol=0)
    torch.testing.assert_close(x, torch.zeros_like(x))


def test_run_example_backward_compares_gradients() -> None:
    x = torch.randn(4, requires_grad=True)

    run_example(lambda t: t.square(), lambda t: t * t, (x,), bwd=True, benchmark=False)


def test_run_example_backward_grad_presence_mismatch_raises() -> None:
    x = torch.randn(4, requires_grad=True)

    def detached_leaf(t: torch.Tensor) -> torch.Tensor:
        return t.square().detach().clone().requires_grad_()

    with pytest.raises(AssertionError, match="kernel: grad presence mismatch for arg 0"):
        run_example(detached_leaf, lambda t: t.square(), (x,), bwd=True, benchmark=False)


def test_run_example_backward_gradient_value_mismatch_raises() -> None:
    x = torch.randn(4, requires_grad=True)

    def same_forward_different_backward(t: torch.Tensor) -> torch.Tensor:
        return t.detach() + 2 * (t - t.detach())

    with pytest.raises(AssertionError, match="kernel: gradient for arg 0"):
        run_example(
            same_forward_different_backward,
            lambda t: t,
            (x,),
            bwd=True,
            benchmark=False,
            rtol=0,
            atol=0,
        )


def test_run_example_backward_requires_grad_input() -> None:
    with pytest.raises(AssertionError, match="no arg has requires_grad"):
        run_example(lambda t: t, lambda t: t, (torch.randn(4),), bwd=True, benchmark=False)


def test_find_free_port_returns_bindable_port() -> None:
    port = _find_free_port()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))


@pytest.mark.parametrize("world_size", [1, 2])
def test_run_distributed_smoke(world_size: int) -> None:
    run_distributed(_noop_worker, world_size=world_size)


def test_run_distributed_worker_assertion_propagates() -> None:
    with pytest.raises(ProcessRaisedException, match="failed intentionally"):
        run_distributed(_assertion_failure_worker, world_size=1)


def test_run_distributed_forwards_args_and_kwargs() -> None:
    run_distributed(
        _args_kwargs_worker,
        world_size=1,
        args=("forwarded",),
        kwargs={"expected_kwarg": 7},
    )


def test_run_distributed_sets_rank_environment() -> None:
    run_distributed(_env_worker, world_size=2)


def test_run_distributed_tears_down_after_failure() -> None:
    with pytest.raises(ProcessRaisedException):
        run_distributed(_assertion_failure_worker, world_size=1)

    run_distributed(_noop_worker, world_size=1)


def test_run_distributed_barrier_allows_uneven_worker_timing() -> None:
    run_distributed(_uneven_worker, world_size=2)


@requires_cuda
def test_do_bench_returns_float() -> None:
    from testkit import do_bench

    x = torch.randn(128, device="cuda")

    assert isinstance(do_bench(lambda: x + 1), float)


def test_print_table_marks_reference_baseline(capsys: pytest.CaptureFixture[str]) -> None:
    _print_table({"kernel": 2.0, "ref": 1.0}, {"ref": object()})

    assert "1.00x (ref)" in capsys.readouterr().err
