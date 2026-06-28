import math
from typing import Any

import pytest
import torch
from torch.optim.adam import adam
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

import dtensor_muon.optim.optim as optim_module
from dtensor_muon.optim.optim import Muon
from testkit import run_distributed

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_compile_false_uses_private_uncompiled_step_callables():
    p = torch.nn.Parameter(torch.randn(2, 2))

    optimizer = Muon([p], compile=False)

    assert "adam" not in optimizer.__dict__
    assert "muon" not in optimizer.__dict__
    assert optimizer._adam_impl is adam
    assert optimizer._muon_impl == optimizer.muon


def test_compile_true_compiles_step_callables_without_shadowing_public_methods(monkeypatch):
    compiled = []

    def fake_compile(fn, *, dynamic):
        compiled.append((fn, dynamic))

        def compiled_fn(*args, **kwargs):
            return fn(*args, **kwargs)

        return compiled_fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    p = torch.nn.Parameter(torch.randn(2, 2))

    optimizer = Muon([p], compile=True)

    assert "adam" not in optimizer.__dict__
    assert "muon" not in optimizer.__dict__
    assert optimizer._adam_impl is not adam
    assert optimizer._muon_impl != optimizer.muon
    assert compiled == [(adam, True), (optimizer.muon, True)]


@requires_cuda
def test_compile_true_real_compiled_muon_group_steps_on_cuda():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(8, 4, device="cuda"))
    before = p.detach().clone()
    p.grad = torch.randn_like(p)
    optimizer = Muon(
        [p],
        compile=True,
        lr=0.01,
        wd=0.0,
        momentum=0.0,
        nesterov=False,
        orthogonalization_strategy="newton_schulz",
    )

    optimizer.step()

    assert not torch.equal(p, before)
    assert torch.equal(optimizer.state[p]["step"], torch.tensor(1.0))


@requires_cuda
def test_compile_true_real_compiled_adam_group_steps_on_cuda():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(4, device="cuda"))
    before = p.detach().clone()
    p.grad = torch.randn_like(p)
    optimizer = Muon(
        [{"params": [p], "algorithm": "adamw", "lr": 0.01, "wd": 0.0}],
        compile=True,
    )

    optimizer.step()

    assert not torch.equal(p, before)
    assert torch.equal(optimizer.state[p]["step"], torch.tensor(1.0, device=p.device))


def _compiled_adamw_dtensor_worker(rank: int, world_size: int) -> None:
    mesh = init_device_mesh("cpu", (world_size,))
    torch.manual_seed(1000 + rank)
    full_param = torch.randn(8 * world_size, 4)
    full_grad = torch.randn_like(full_param)
    param = torch.nn.Parameter(distribute_tensor(full_param, mesh, [Shard(0)]))
    param.grad = distribute_tensor(full_grad, mesh, [Shard(0)])
    optimizer = Muon(
        [
            {
                "params": [param],
                "algorithm": "adamw",
                "lr": 0.01,
                "wd": 0.0,
                "fused": False,
            }
        ]
    )

    @torch.compile(dynamic=False)
    def compiled_step(opt):
        opt.step()

    compiled_step(optimizer)

    assert torch.equal(optimizer.state[param]["step"], torch.tensor(1.0))
    assert torch.isfinite(param.full_tensor()).all()


def test_compiled_adamw_group_steps_dtensor_params():
    run_distributed(_compiled_adamw_dtensor_worker, world_size=2)


@requires_cuda
def test_fused_adam_cuda_state_step_is_on_param_device():
    p = torch.nn.Parameter(torch.randn(4, device="cuda"))
    optimizer = Muon([{"params": [p], "algorithm": "adamw", "fused": True}])
    p.grad = torch.randn_like(p)

    optimizer.step()

    assert optimizer.state[p]["step"].device == p.device
    assert optimizer.state[p]["step"].dtype is torch.float32


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lr": -1.0}, "Invalid learning rate"),
        ({"adam_eps": -1.0}, "Invalid epsilon"),
        ({"adam_betas": (1.0, 0.9)}, "index 0"),
        ({"adam_betas": (0.9, -0.1)}, "index 1"),
    ],
)
def test_constructor_validates_scalar_hyperparameters(kwargs, match):
    p = torch.nn.Parameter(torch.randn(2, 2))

    with pytest.raises(ValueError, match=match):
        Muon([p], **kwargs)


def test_constructor_rejects_single_param_group_dict():
    p = torch.nn.Parameter(torch.randn(2, 2))

    with pytest.raises(TypeError, match="not a single dict"):
        Muon({"params": [p]})


def test_constructor_rejects_empty_params():
    with pytest.raises(ValueError, match="empty parameter list"):
        Muon([])


def test_constructor_rejects_group_missing_params_key():
    with pytest.raises(ValueError, match="'params' key"):
        Muon([{"algorithm": "muon"}])


def test_constructor_skips_empty_groups_when_other_groups_exist():
    p = torch.nn.Parameter(torch.randn(2, 2))

    optimizer = Muon([{"params": []}, {"params": [p]}])

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["params"] == [p]


def test_constructor_rejects_unknown_algorithm():
    p = torch.nn.Parameter(torch.randn(2, 2))

    with pytest.raises(ValueError, match="Unknown algorithm"):
        Muon([{"params": [p], "algorithm": "rmsprop"}])


def test_constructor_rejects_invalid_muon_parameters():
    one_d = torch.nn.Parameter(torch.randn(4))
    complex_param = torch.nn.Parameter(torch.randn(2, 2, dtype=torch.cfloat))

    with pytest.raises(ValueError, match="2D\\+"):
        Muon([one_d])
    with pytest.raises(NotImplementedError, match="Complex parameters"):
        Muon([complex_param])


def test_constructor_normalizes_muon_and_adam_groups():
    muon_param = torch.nn.Parameter(torch.randn(2, 2))
    adam_param = torch.nn.Parameter(torch.randn(4))

    optimizer = Muon(
        [
            {"params": [muon_param], "lr": 0.5, "wd": 0.25},
            {"params": [adam_param], "algorithm": "adamw", "lr": 0.3, "wd": 0.1},
        ],
        lr=0.01,
        wd=0.9,
    )

    muon_group, adam_group = optimizer.param_groups
    assert muon_group["use_muon"] is True
    assert torch.equal(muon_group["lr"], torch.tensor(0.5))
    assert muon_group["wd"] == 0.25
    assert adam_group["use_muon"] is False
    assert torch.equal(adam_group["lr"], torch.tensor(0.3))
    assert torch.equal(adam_group["wd"], torch.tensor(0.1))


def test_constructor_allows_3d_muon_and_complex_adam_parameters():
    muon_param = torch.nn.Parameter(torch.randn(2, 3, 4))
    adam_param = torch.nn.Parameter(torch.randn(2, 2, dtype=torch.cfloat))

    optimizer = Muon(
        [
            {"params": [muon_param]},
            {"params": [adam_param], "algorithm": "adam", "fused": False},
        ]
    )

    assert optimizer.param_groups[0]["use_muon"] is True
    assert optimizer.param_groups[1]["has_complex"] is True


def test_adam_group_matches_torch_adamw_on_cpu():
    p = torch.nn.Parameter(torch.randn(3, 4))
    ref = torch.nn.Parameter(p.detach().clone())
    optimizer = Muon(
        [
            {
                "params": [p],
                "algorithm": "adamw",
                "lr": 0.01,
                "wd": 0.1,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "fused": False,
            }
        ]
    )
    ref_optimizer = torch.optim.AdamW(
        [ref], lr=0.01, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8
    )

    for _ in range(3):
        grad = torch.randn_like(p)
        p.grad = grad.clone()
        ref.grad = grad.clone()
        optimizer.step()
        ref_optimizer.step()

    torch.testing.assert_close(p, ref)


def test_algorithm_adam_matches_torch_adam_coupled_weight_decay_on_cpu():
    p = torch.nn.Parameter(torch.randn(3, 4))
    ref = torch.nn.Parameter(p.detach().clone())
    optimizer = Muon(
        [
            {
                "params": [p],
                "algorithm": "adam",
                "lr": 0.01,
                "wd": 0.1,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "fused": False,
            }
        ]
    )
    ref_optimizer = torch.optim.Adam(
        [ref], lr=0.01, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8
    )

    for _ in range(3):
        grad = torch.randn_like(p)
        p.grad = grad.clone()
        ref.grad = grad.clone()
        optimizer.step()
        ref_optimizer.step()

    torch.testing.assert_close(p, ref)


def test_adam_group_amsgrad_matches_torch_adamw_on_cpu():
    p = torch.nn.Parameter(torch.randn(3, 4))
    ref = torch.nn.Parameter(p.detach().clone())
    optimizer = Muon(
        [
            {
                "params": [p],
                "algorithm": "adamw",
                "amsgrad": True,
                "lr": 0.01,
                "wd": 0.1,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "fused": False,
            }
        ]
    )
    ref_optimizer = torch.optim.AdamW(
        [ref],
        lr=0.01,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        amsgrad=True,
    )

    for _ in range(3):
        grad = torch.randn_like(p)
        p.grad = grad.clone()
        ref.grad = grad.clone()
        optimizer.step()
        ref_optimizer.step()

    assert "max_exp_avg_sq" in optimizer.state[p]
    torch.testing.assert_close(p, ref)


def test_complex_adam_group_steps_on_cpu():
    p = torch.nn.Parameter(torch.randn(2, 2, dtype=torch.cfloat))
    optimizer = Muon([{"params": [p], "algorithm": "adam", "fused": False}])
    p.grad = torch.randn_like(p)

    optimizer.step()

    assert optimizer.state[p]["step"].item() == 1


def test_step_returns_closure_loss():
    p = torch.nn.Parameter(torch.randn(3, 4))
    optimizer = Muon([{"params": [p], "algorithm": "adamw", "fused": False}])

    def closure():
        return p.square().sum()

    assert optimizer.step(closure) == closure()


def test_muon_direct_update_initializes_state_and_uses_shape_lr_ratio(monkeypatch):
    monkeypatch.setattr(optim_module, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.ones(4, 2))
    p.grad = torch.full_like(p, 0.5)
    optimizer = Muon([p], lr=0.1, wd=0.0, momentum=0.0, nesterov=False)

    optimizer.step()

    state = optimizer.state[p]
    assert torch.equal(state["step"], torch.tensor(1.0))
    assert torch.equal(state["lr_ratio"], torch.tensor(math.sqrt(2.0)))
    torch.testing.assert_close(state["momentum_buffer"], torch.full_like(p, 0.5))
    torch.testing.assert_close(p, torch.ones_like(p) - math.sqrt(2.0) * 0.1 * 0.5)


def test_muon_nesterov_uses_updated_momentum_buffer(monkeypatch):
    monkeypatch.setattr(optim_module, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.ones(2, 2))
    p.grad = torch.full_like(p, 0.5)
    optimizer = Muon([p], lr=0.1, wd=0.0, momentum=0.5, nesterov=True)

    optimizer.step()

    # buf = grad; nesterov update = grad + momentum * buf = 0.75
    torch.testing.assert_close(optimizer.state[p]["momentum_buffer"], torch.full_like(p, 0.5))
    torch.testing.assert_close(p, torch.ones_like(p) - 0.1 * 0.75)


def test_muon_cautious_weight_decay_masks_by_update_param_sign(monkeypatch):
    monkeypatch.setattr(optim_module, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.tensor([[1.0, -1.0], [1.0, -1.0]]))
    p.grad = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    optimizer = Muon(
        [p],
        lr=1.0,
        wd=1.0,
        momentum=0.0,
        nesterov=False,
        use_cautious_wd=True,
    )

    optimizer.step()

    torch.testing.assert_close(p, torch.tensor([[-1.0, -2.0], [2.0, 1.0]]))


def test_muon_non_cautious_weight_decay_adds_param_everywhere(monkeypatch):
    monkeypatch.setattr(optim_module, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.tensor([[1.0, -1.0], [1.0, -1.0]]))
    p.grad = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    optimizer = Muon(
        [p],
        lr=1.0,
        wd=1.0,
        momentum=0.0,
        nesterov=False,
        use_cautious_wd=False,
    )

    optimizer.step()

    torch.testing.assert_close(p, torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))


def test_muon_maximize_negates_grad_in_place(monkeypatch):
    monkeypatch.setattr(optim_module, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.ones(2, 2))
    grad = torch.full_like(p, 0.5)
    p.grad = grad
    optimizer = Muon([p], lr=0.1, wd=0.0, momentum=0.0, nesterov=False, maximize=True)

    optimizer.step()

    torch.testing.assert_close(grad, torch.full_like(grad, 0.5))
    torch.testing.assert_close(p, torch.full_like(p, 1.05))


def test_muon_maximize_reused_grad_keeps_maximize_direction(monkeypatch):
    monkeypatch.setattr(optim_module, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.ones(2, 2))
    grad = torch.full_like(p, 0.5)
    p.grad = grad
    optimizer = Muon([p], lr=0.1, wd=0.0, momentum=0.0, nesterov=False, maximize=True)

    optimizer.step()
    after_first = p.detach().clone()
    optimizer.step()

    torch.testing.assert_close(grad, torch.full_like(grad, 0.5))
    torch.testing.assert_close(after_first, torch.full_like(after_first, 1.05))
    torch.testing.assert_close(p, torch.full_like(p, 1.10))


def test_muon_momentum_buffer_uses_fp32_for_low_precision_params(monkeypatch):
    monkeypatch.setattr(optim_module, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))
    p.grad = torch.full_like(p, 0.5)
    optimizer = Muon([p], lr=0.1, wd=0.0, momentum=0.9, nesterov=False)

    optimizer.step()

    assert optimizer.state[p]["momentum_buffer"].dtype is torch.float32


def test_muon_skips_params_without_grad(monkeypatch):
    monkeypatch.setattr(optim_module, "zeropower", lambda g, **_: g)
    with_grad = torch.nn.Parameter(torch.ones(2, 2))
    without_grad = torch.nn.Parameter(torch.ones(2, 2))
    with_grad.grad = torch.full_like(with_grad, 0.5)
    optimizer = Muon([with_grad, without_grad], lr=0.1, wd=0.0, momentum=0.0)

    optimizer.step()

    assert with_grad in optimizer.state
    assert without_grad not in optimizer.state
    torch.testing.assert_close(without_grad, torch.ones_like(without_grad))


def test_muon_flatten_true_updates_ndim_greater_than_three(monkeypatch):
    seen_shapes = []

    def fake_zeropower(g, **_):
        seen_shapes.append(tuple(g.shape))
        return g

    monkeypatch.setattr(optim_module, "zeropower", fake_zeropower)
    p = torch.nn.Parameter(torch.ones(2, 2, 2, 2))
    p.grad = torch.full_like(p, 0.5)
    optimizer = Muon([{"params": [p], "flatten": True}], lr=0.1, wd=0.0, momentum=0.0)

    optimizer.step()

    assert seen_shapes == [(2, 8)]
    assert torch.equal(optimizer.state[p]["lr_ratio"], torch.tensor(1.0))
    torch.testing.assert_close(p, torch.ones_like(p) - 0.1 * 0.5)


def test_muon_flatten_false_rejects_ndim_greater_than_three(monkeypatch):
    monkeypatch.setattr(optim_module, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.ones(2, 2, 2, 2))
    p.grad = torch.full_like(p, 0.5)
    optimizer = Muon([{"params": [p], "flatten": False}])

    with pytest.raises(AssertionError, match="Please set flatten=True"):
        optimizer.step()


def test_muon_flatten_false_steps_3d_grad_without_flattening(monkeypatch):
    seen_shapes = []

    def fake_zeropower(g, **_):
        seen_shapes.append(tuple(g.shape))
        return g

    monkeypatch.setattr(optim_module, "zeropower", fake_zeropower)
    p = torch.nn.Parameter(torch.ones(2, 2, 2))
    p.grad = torch.full_like(p, 0.5)
    optimizer = Muon([{"params": [p], "flatten": False}], lr=0.1, wd=0.0, momentum=0.0)

    optimizer.step()

    assert seen_shapes == [(2, 2, 2)]
    torch.testing.assert_close(p, torch.ones_like(p) - 0.1 * 0.5)


def test_load_state_dict_applies_legacy_defaults_and_tensor_steps():
    p = torch.nn.Parameter(torch.randn(2, 2))
    optimizer = Muon([p])
    group = optimizer.param_groups[0]
    for key in [
        "ns_steps",
        "nesterov",
        "flatten",
        "use_cautious_wd",
        "orthogonalization_strategy",
    ]:
        del group[key]
    group["lr"] = 0.1
    optimizer.state[p]["step"] = 3.0

    optimizer.__setstate__({"state": optimizer.state, "param_groups": optimizer.param_groups})

    assert group["ns_steps"] == 5
    assert group["nesterov"] is True
    assert group["flatten"] is True
    assert group["use_cautious_wd"] is True
    # Legacy default matches the constructor default ("polar_express"), so a
    # checkpoint missing the key doesn't silently switch the orthogonalization scheme.
    assert group["orthogonalization_strategy"] == "polar_express"
    lr = group["lr"]
    assert isinstance(lr, torch.Tensor)
    assert torch.equal(lr, torch.tensor(0.1))
    assert torch.equal(optimizer.state[p]["step"], torch.tensor(3.0))


def test_load_state_dict_applies_adam_legacy_defaults_and_step_tensor():
    p = torch.nn.Parameter(torch.randn(2, 2))
    optimizer = Muon([{"params": [p], "algorithm": "adamw", "fused": False}])
    group = optimizer.param_groups[0]
    for key in ["amsgrad", "maximize", "foreach", "decoupled_weight_decay", "fused"]:
        group.pop(key, None)
    optimizer.state[p]["step"] = 4.0

    optimizer.__setstate__({"state": optimizer.state, "param_groups": optimizer.param_groups})

    assert group["amsgrad"] is False
    assert group["maximize"] is False
    assert group["foreach"] is None
    assert group["decoupled_weight_decay"] is False
    assert group["fused"] is None
    assert torch.equal(optimizer.state[p]["step"], torch.tensor(4.0))


def test_setstate_reinitializes_compiled_step_impls(monkeypatch):
    compiled = []

    def fake_compile(fn, *, dynamic):
        compiled.append((fn, dynamic))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    p = torch.nn.Parameter(torch.randn(2, 2))
    optimizer = Muon([p], compile=True)
    compiled.clear()

    optimizer.__setstate__({"state": optimizer.state, "param_groups": optimizer.param_groups})

    assert compiled == [(adam, True), (optimizer.muon, True)]


def test_state_dict_round_trip_preserves_muon_training_continuity(monkeypatch):
    monkeypatch.setattr(optim_module, "zeropower", lambda g, **_: g)
    p = torch.nn.Parameter(torch.ones(4, 2))
    uninterrupted_p = torch.nn.Parameter(p.detach().clone())
    loaded_p = torch.nn.Parameter(p.detach().clone())
    kwargs: dict[str, Any] = dict(lr=0.1, wd=0.0, momentum=0.5, nesterov=False)
    uninterrupted = Muon([uninterrupted_p], **kwargs)
    to_save = Muon([p], **kwargs)
    loaded = Muon([loaded_p], **kwargs)

    first_grad = torch.full_like(p, 0.5)
    for param in (uninterrupted_p, p):
        param.grad = first_grad.clone()
    uninterrupted.step()
    to_save.step()
    loaded_p.data.copy_(p.data)
    loaded.load_state_dict(to_save.state_dict())

    second_grad = torch.full_like(p, 0.25)
    for param in (uninterrupted_p, loaded_p):
        param.grad = second_grad.clone()
    uninterrupted.step()
    loaded.step()

    torch.testing.assert_close(loaded_p, uninterrupted_p)
    for key in ["step", "momentum_buffer", "lr_ratio"]:
        torch.testing.assert_close(loaded.state[loaded_p][key], uninterrupted.state[uninterrupted_p][key])


def test_mixed_muon_and_adam_groups_step_on_cpu(monkeypatch):
    monkeypatch.setattr(optim_module, "zeropower", lambda g, **_: g)
    muon_param = torch.nn.Parameter(torch.ones(2, 2))
    adam_param = torch.nn.Parameter(torch.randn(3))
    adam_ref = torch.nn.Parameter(adam_param.detach().clone())
    optimizer = Muon(
        [
            {"params": [muon_param], "lr": 0.1, "wd": 0.0, "momentum": 0.0},
            {
                "params": [adam_param],
                "algorithm": "adamw",
                "lr": 0.01,
                "wd": 0.1,
                "fused": False,
            },
        ]
    )
    ref_optimizer = torch.optim.AdamW([adam_ref], lr=0.01, weight_decay=0.1)

    muon_param.grad = torch.full_like(muon_param, 0.5)
    adam_grad = torch.randn_like(adam_param)
    adam_param.grad = adam_grad.clone()
    adam_ref.grad = adam_grad.clone()

    optimizer.step()
    ref_optimizer.step()

    torch.testing.assert_close(muon_param, torch.full_like(muon_param, 0.95))
    torch.testing.assert_close(adam_param, adam_ref)
