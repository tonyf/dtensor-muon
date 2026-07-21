import functools
import math
from typing import cast

import torch
from torch import Tensor
from torch.optim.adam import adam
from torch.optim.optimizer import (
    _device_dtype_check_for_fused,
    _get_scalar_dtype,
    _use_grad_for_differentiable,
)

from muonium.orthogonalize import OrthogonalizationStrategy
from muonium.utils import group_tensors_by_shape, move_tensors_to_device

from .algorithms import BufferSpec, MuonAlgorithm, get_algorithm
from .algorithms.base import ADAM_ALGORITHMS


def _register_dtensor_foreach_ops():
    """Register _foreach_sign_ with DTensor using official register_op_strategy API."""
    from torch.distributed.tensor._op_schema import (
        OpSpec,
        OpStrategy,
        RuntimeSchemaInfo,
        TupleStrategy,
    )
    from torch.distributed.tensor._ops.utils import register_op_strategy

    op = torch.ops.aten._foreach_sign_.default
    if hasattr(op, "_dtensor_registered"):
        return

    @register_op_strategy(op, schema_info=RuntimeSchemaInfo(needs_pytree=True))
    def _(op_schema):
        arg = op_schema.args_schema[0]
        return TupleStrategy(
            [
                OpStrategy([OpSpec(s.output_specs, (s.output_specs,)) for s in c.strategies])
                for c in arg.children
            ]
        )

    setattr(op, "_dtensor_registered", True)


_register_dtensor_foreach_ops()


def _group_muon_tensors(
    params: list[Tensor],
    grads: list[Tensor],
    state_lists: dict[str, list[Tensor]],
    lr_ratios: list[Tensor],
) -> list[
    tuple[torch.device, list[Tensor], list[Tensor], dict[str, list[Tensor]], list[Tensor]]
]:
    """Build static homogeneous batches that are safe to stack during compilation."""
    groups: dict[tuple[torch.device, torch.dtype], list[int]] = {}
    for idx, param in enumerate(params):
        groups.setdefault((param.device, param.dtype), []).append(idx)

    homogeneous_groups = []
    for (device, _), group_indices in groups.items():
        group_grads = [grads[i] for i in group_indices]
        for _, (_, shape_indices) in group_tensors_by_shape(group_grads).items():
            indices = [group_indices[i] for i in shape_indices]
            homogeneous_groups.append(
                (
                    device,
                    [params[i] for i in indices],
                    [grads[i] for i in indices],
                    {key: [ts[i] for i in indices] for key, ts in state_lists.items()},
                    [lr_ratios[i] for i in indices],
                )
            )
    return homogeneous_groups


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        wd: float = 0.1,
        use_cautious_wd: bool = True,
        maximize: bool = False,
        # Muon defaults
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        orthogonalization_strategy: OrthogonalizationStrategy = "polar_express",
        # Adam defaults
        amsgrad: bool = False,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        is_adamw: bool = True,
        fused_adam: bool | None = None,
        foreach_adam: bool | None = None,
        # Execution strategy
        foreach: bool | None = None,
        batch_size: int | None = None,
    ):
        """
        Unified Muon + Adam optimizer.

        Args:
            params: An iterable of parameters or an iterable of param-group dicts.
                Each param-group dict can have an optional "algorithm" key:
                - "adamw" or "adam": Use Adam/AdamW for this group
                - "muon" or omitted: Use Muon for this group (default)
                - any registered Muon variant name (e.g. "normuon"); see
                  ``muonium.register_muon_algorithm``. Variant-specific
                  hyperparameters (e.g. NorMuon's ``muon_beta2``) are set in the
                  group dict alongside ``algorithm``.
                Muon-family groups may also set:
                - ``flatten`` (default False): whether to flatten 3D+ tensors to
                  2D for the Muon update. False treats them as batches of 2D
                  matrices, each orthogonalized independently (leading dims fold
                  into the batch); True collapses them to one ``(dim0, -1)``
                  matrix — use this for convolutional layers.
                - ``split_sizes`` (2D params only): orthogonalize row blocks of
                  a fused weight (e.g. QKV) independently, as separate
                  parameters would be.
            lr: Default learning rate
            wd: Default weight decay
            use_cautious_wd: Use cautious weight decay for Muon groups. When enabled,
                weight decay is only applied when the update and parameter have the
                same sign (i.e., when u * p > 0). This is an addition to the original
                Muon, which has no cautious variant; set to False for plain
                (non-cautious) decoupled weight decay.
            momentum: Muon momentum
            nesterov: Use Nesterov momentum for Muon
            ns_steps: Number of orthogonalization steps for Muon
            orthogonalization_strategy: Algorithm for computing zero-power/sign of gradients:
                - "newton_schulz": Classic Newton-Schulz iteration (default)
                - "polar_express": Polar Express algorithm (arxiv.org/pdf/2505.16932)
            amsgrad: Use AMSGrad for Adam
            adam_betas: Beta parameters for Adam
            adam_eps: Epsilon for Adam
            is_adamw: Use decoupled weight decay (AdamW) vs coupled (Adam)
            fused_adam: Use fused Adam kernel
            maximize: Maximize instead of minimize
            foreach: Batch the Muon-family update with ``torch._foreach_*`` ops,
                grouping params by (device, dtype, shape). ``None`` (default)
                enables it per group when every param lives on CUDA. Explicit
                ``True`` also opts CPU(-offloaded) groups into a CUDA round trip
                per batch; requires CUDA. Overridable per param group (the group
                key also controls foreach ``zero_grad``, as in torch.optim).
            batch_size: Maximum tensors per foreach batch (None = unbounded).
        """
        self.lr = lr
        self.wd = wd
        self.use_cautious_wd = use_cautious_wd
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.orthogonalization_strategy = orthogonalization_strategy
        self.amsgrad = amsgrad
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps
        self.is_adamw = is_adamw
        self.fused_adam = fused_adam
        self.foreach_adam = foreach_adam
        self.maximize = maximize
        self.foreach = foreach
        self.batch_size = batch_size

        param_groups: list[dict] = []

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= adam_eps:
            raise ValueError(f"Invalid epsilon value: {adam_eps}")
        if not 0.0 <= adam_betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {adam_betas[0]}")
        if not 0.0 <= adam_betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {adam_betas[1]}")

        if isinstance(params, dict):
            raise TypeError(
                "params must be an iterable of parameters or "
                "an iterable of param-group dicts, not a single dict."
            )

        params = list(params)
        if len(params) == 0:
            raise ValueError("optimizer got an empty parameter list")

        if not isinstance(params[0], dict):
            params = [{"params": params}]

        for group in params:
            algorithm = cast(str, group.get("algorithm", "muon")).lower()
            if "params" not in group:
                raise ValueError("Param group dict must have a 'params' key")

            group["params"] = list(group["params"])
            if len(group["params"]) == 0:
                continue

            if algorithm in ADAM_ALGORITHMS:
                param_groups.append(self._build_adam_group(group))
            else:
                param_groups.append(self._build_muon_group(group, get_algorithm(algorithm)))

        super().__init__(param_groups, {"differentiable": False})
        self._init_muon_impl()

    def _init_muon_impl(self) -> None:
        """Reset the per-algorithm cache of compiled Muon kernels.

        Kernels are compiled lazily on first use (per algorithm name) so groups
        added after construction — or loaded from checkpoints — get their own
        compiled entry.
        """
        self._muon_impls: dict[str, object] = {}

    def _raw_muon_impl(self, algorithm: MuonAlgorithm):
        """The uncompiled update entry point with the algorithm bound."""
        return functools.partial(self.muon, algorithm)

    def _get_muon_impl(self, algorithm: MuonAlgorithm):
        impl = self._muon_impls.get(algorithm.name)
        if impl is None:
            impl = torch.compile(self._raw_muon_impl(algorithm), dynamic=True)
            self._muon_impls[algorithm.name] = impl
        return impl

    @staticmethod
    def _validate_split_sizes(group: dict) -> tuple[int, ...] | None:
        """Validate the optional ``split_sizes`` group option (2D fused weights)."""
        split_sizes = group.get("split_sizes")
        if split_sizes is None:
            return None
        if not isinstance(split_sizes, (tuple, list)) or len(split_sizes) < 2:
            raise ValueError(
                f"split_sizes must be a tuple or list of at least 2 block sizes, "
                f"got {split_sizes!r}."
            )
        if not all(isinstance(s, int) and not isinstance(s, bool) and s > 0 for s in split_sizes):
            raise ValueError(
                f"split_sizes entries must be positive integers, got {split_sizes!r}."
            )
        for p in group["params"]:
            if p.ndim != 2:
                raise ValueError(
                    f"split_sizes is only supported for 2D parameters, got shape "
                    f"{tuple(p.shape)}."
                )
            if sum(split_sizes) != p.shape[0]:
                raise ValueError(
                    f"split_sizes {tuple(split_sizes)} must sum to dim 0 of the "
                    f"parameter shape {tuple(p.shape)}."
                )
        return tuple(split_sizes)

    def _build_muon_group(self, group: dict, algorithm: MuonAlgorithm):
        for p in group["params"]:
            algorithm.validate_param(p)

        # Resolve the execution strategy to a concrete bool at build time. Auto
        # (None) matches torch.optim: batch only when every param already lives
        # on CUDA — explicit foreach=True opts CPU(-offloaded) params into the
        # per-batch CUDA round trip.
        foreach = group.get("foreach", self.foreach)
        if foreach is None:
            foreach = all(p.is_cuda for p in group["params"])

        built = {
            "params": group["params"],
            "algorithm": algorithm.name,
            "use_muon": True,
            "lr": torch.tensor(group.get("lr", self.lr)),
            "wd": group.get("wd", self.wd),
            "use_cautious_wd": group.get("use_cautious_wd", self.use_cautious_wd),
            "momentum": group.get("momentum", self.momentum),
            "nesterov": group.get("nesterov", self.nesterov),
            "ns_steps": group.get("ns_steps", self.ns_steps),
            "orthogonalization_strategy": group.get(
                "orthogonalization_strategy", self.orthogonalization_strategy
            ),
            "maximize": group.get("maximize", self.maximize),
            "flatten": group.get("flatten", False),
            "split_sizes": self._validate_split_sizes(group),
            # Selects the batched foreach driver in muon(); also consumed by the
            # base Optimizer's foreach zero_grad, as in torch.optim.
            "foreach": foreach,
        }
        # Variant-specific hyperparameters travel in the group dict; unset ones
        # fall back to the algorithm's declared defaults.
        for key, default in algorithm.options.items():
            built[key] = group.get(key, default)
        return built

    def _build_adam_group(self, group: dict):
        algorithm = cast(str, group.get("algorithm", "adamw")).lower()
        return {
            "params": group["params"],
            "algorithm": algorithm,
            "use_muon": False,
            "lr": group.get("lr", self.lr),
            "wd": group.get("wd", self.wd),
            "amsgrad": group.get("amsgrad", self.amsgrad),
            "betas": group.get("betas", self.adam_betas),
            "eps": group.get("eps", self.adam_eps),
            "decoupled_weight_decay": group.get(
                "decoupled_weight_decay", self.is_adamw and algorithm == "adamw"
            ),
            "fused": group.get("fused", self.fused_adam),
            "foreach": group.get("foreach", self.foreach_adam),
            "maximize": group.get("maximize", self.maximize),
            "has_complex": any(torch.is_complex(p) for p in group["params"]),
        }

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            # Muon state initialization
            if group["use_muon"]:
                group.setdefault("algorithm", "muon")
                algorithm = get_algorithm(group["algorithm"])
                group.setdefault("ns_steps", 5)
                group.setdefault("nesterov", True)
                group.setdefault("flatten", False)
                group.setdefault("use_cautious_wd", True)
                group.setdefault("orthogonalization_strategy", "polar_express")
                group.setdefault("split_sizes", None)
                if "foreach" not in group:
                    group["foreach"] = all(p.is_cuda for p in group["params"])
                for key, default in algorithm.options.items():
                    group.setdefault(key, default)
                if not torch.is_tensor(group["lr"]):
                    group["lr"] = torch.tensor(float(group["lr"]))

                for p in group["params"]:
                    p_state = self.state.get(p, [])
                    if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                        step_val = float(p_state["step"])
                        p_state["step"] = torch.tensor(step_val)

            # Adam state initialization
            else:
                group.setdefault("amsgrad", False)
                group.setdefault("maximize", False)
                group.setdefault("foreach", None)
                group.setdefault("decoupled_weight_decay", False)
                if torch.is_tensor(group["wd"]):
                    group["wd"] = float(group["wd"])
                fused = group.setdefault("fused", None)

                for p in group["params"]:
                    p_state = self.state.get(p, [])
                    if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                        step_val = float(p_state["step"])
                        p_state["step"] = (
                            torch.tensor(
                                step_val,
                                dtype=_get_scalar_dtype(is_fused=fused),
                                device=p.device,
                            )
                            if group["fused"]
                            else torch.tensor(step_val, dtype=_get_scalar_dtype())
                        )

        self._init_muon_impl()

    def _new_state_buffer(self, p: Tensor, grad: Tensor, spec: BufferSpec) -> Tensor:
        """Allocate one per-parameter state buffer declared by an algorithm's
        ``state_spec``. Overridden by low-precision optimizers to quantize."""
        reference = grad if spec.like == "grad" else grad[..., :1]
        return torch.zeros_like(
            reference, dtype=torch.float32, memory_format=torch.preserve_format
        )

    def _init_muon_group(
        self,
        group,
        algorithm: MuonAlgorithm,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        state_lists: dict[str, list[Tensor]],
        lr_ratios: list[Tensor],
        state_steps: list[Tensor],
    ) -> None:
        for p in group["params"]:
            grad = p.grad
            if grad is None:
                continue

            params_with_grad.append(p)
            assert not grad.is_sparse, "Muon does not support sparse gradients"
            if grad.ndim > 2:
                if group["flatten"]:
                    # Opt-in (e.g. conv weights): collapse to a single 2D matrix.
                    grad = grad.view(grad.size(0), -1)
                elif grad.ndim > 3:
                    # Batches of 2D matrices: fold leading dims into one batch dim.
                    grad = grad.flatten(end_dim=-3)

            grads.append(grad)
            state = self.state[p]

            # Lazy state initialization
            if len(state) == 0:
                state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                for key, spec in algorithm.state_spec.items():
                    state[key] = self._new_state_buffer(p, grad, spec)
                state["lr_ratio"] = torch.tensor(
                    math.sqrt(max(1.0, grad.shape[-2] / grad.shape[-1]))
                )

            state["step"] += 1
            for key in algorithm.state_spec:
                state_lists[key].append(state[key])
            lr_ratios.append(state["lr_ratio"])
            state_steps.append(state["step"])

    def _init_adam_group(
        self,
        group,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        max_exp_avg_sqs: list[Tensor],
        state_steps: list[Tensor],
        capturable: bool,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue

            params_with_grad.append(p)
            assert not p.grad.is_sparse, (
                "Adam does not support sparse gradients, please consider SparseAdam instead"
            )

            grads.append(p.grad)
            state = self.state[p]

            # Lazy state initialization
            if len(state) == 0:
                if group["fused"]:
                    _device_dtype_check_for_fused(p)

                # note(crcrpar): [special device hosting for step]
                # Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = (
                    torch.zeros(
                        (),
                        dtype=_get_scalar_dtype(is_fused=group["fused"]),
                        device=p.device,
                    )
                    if capturable or group["fused"]
                    else torch.tensor(0.0, dtype=_get_scalar_dtype())
                )

                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if group["amsgrad"]:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
            elif capturable and state["step"].device != p.device:
                state["step"] = state["step"].to(p.device)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            if group["amsgrad"]:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])

            state_steps.append(state["step"])

    def muon(
        self,
        algorithm: MuonAlgorithm,
        params: list[Tensor],
        grads: list[Tensor],
        state: dict[str, list[Tensor]],
        lr_ratios: list[Tensor],
        *,
        foreach: bool = False,
        nesterov: bool,
        lr: Tensor,
        weight_decay: float,
        cautious_wd: bool,
        momentum: float,
        ns_steps: int,
        orthogonalization_strategy: OrthogonalizationStrategy,
        maximize: bool,
        split_sizes: tuple[int, ...] | None = None,
        **opts,
    ) -> None:
        """
        Muon-family update for a list of parameters.

        Two drivers, selected by ``foreach``: the per-parameter reference loop
        (``algorithm.update`` per tensor), or the batched shell — group tensors
        by (device, dtype, shape), chunk by ``batch_size``, move CPU-offloaded
        batches to CUDA and back — delegating to ``algorithm.foreach_update``.
        The math itself lives on the :class:`MuonAlgorithm` either way.
        """
        if len(params) == 0:
            return

        if not foreach:
            for i, (param, grad, lr_ratio) in enumerate(
                zip(params, grads, lr_ratios, strict=True)
            ):
                algorithm.update(
                    param,
                    grad,
                    {key: buffers[i] for key, buffers in state.items()},
                    lr_ratio,
                    lr=lr,
                    weight_decay=weight_decay,
                    cautious_wd=cautious_wd,
                    momentum=momentum,
                    nesterov=nesterov,
                    maximize=maximize,
                    ns_steps=ns_steps,
                    orthogonalization_strategy=orthogonalization_strategy,
                    split_sizes=split_sizes,
                    **opts,
                )
            return

        for device, device_p, device_g, device_state, device_lr in _group_muon_tensors(
            params, grads, state, lr_ratios
        ):
            indices = list(range(len(device_g)))
            batches = (
                [
                    indices[i : i + self.batch_size]
                    for i in range(0, len(indices), self.batch_size)
                ]
                if self.batch_size and len(indices) > self.batch_size
                else [indices]
            )

            for batch_idx in batches:
                batch_p_orig = [device_p[i] for i in batch_idx]
                batch_g_orig = [device_g[i] for i in batch_idx]
                batch_state_orig = {
                    key: [ts[i] for i in batch_idx] for key, ts in device_state.items()
                }
                batch_lr_orig = [device_lr[i] for i in batch_idx]

                # Move to CUDA for processing (handles CPU offload)
                cuda = torch.device("cuda")
                batch_p = move_tensors_to_device(batch_p_orig, device, cuda)
                batch_g = move_tensors_to_device(batch_g_orig, device, cuda)
                batch_state = {
                    key: move_tensors_to_device(ts, device, cuda)
                    for key, ts in batch_state_orig.items()
                }
                batch_lr = move_tensors_to_device(batch_lr_orig, device, cuda)

                algorithm.foreach_update(
                    cast(list[Tensor], batch_p),
                    cast(list[Tensor], batch_g),
                    cast(dict[str, list[Tensor]], batch_state),
                    cast(list[Tensor], batch_lr),
                    nesterov=nesterov,
                    lr=lr,
                    weight_decay=weight_decay,
                    cautious_wd=cautious_wd,
                    momentum=momentum,
                    maximize=maximize,
                    ns_steps=ns_steps,
                    orthogonalization_strategy=orthogonalization_strategy,
                    split_sizes=split_sizes,
                    **opts,
                )

                # CPU offload mutates CUDA copies; copy those values back to the
                # original tensors. Same-device batches alias and need no copy.
                moved_pairs = [
                    (batch_p_orig, batch_p),
                    (batch_g_orig, batch_g),
                    (batch_lr_orig, batch_lr),
                ]
                moved_pairs.extend(
                    (batch_state_orig[key], batch_state[key]) for key in batch_state_orig
                )
                for originals, moved in moved_pairs:
                    if originals is not moved:
                        for original, value in zip(originals, moved, strict=True):
                            if original is not None and value is not None:
                                original.copy_(value.to(original.device))

    def _step_muon_group(self, group: dict) -> None:
        """Process a single Muon param group."""
        algorithm = get_algorithm(group.get("algorithm", "muon"))
        params_with_grad: list[Tensor] = []
        grads: list[Tensor] = []
        state_steps: list[Tensor] = []
        state_lists: dict[str, list[Tensor]] = {key: [] for key in algorithm.state_spec}
        lr_ratios: list[Tensor] = []

        self._init_muon_group(
            group,
            algorithm,
            params_with_grad,
            grads,
            state_lists,
            lr_ratios,
            state_steps,
        )

        # Standalone steps use the regional compiled kernel for performance. If an
        # outer torch.compile is tracing step(), use the raw method so the update is
        # captured in that optimizer graph instead of entering a nested compiler.
        muon_impl = (
            self._raw_muon_impl(algorithm)
            if torch.compiler.is_compiling()
            else self._get_muon_impl(algorithm)
        )
        muon_impl(
            params_with_grad,
            grads,
            state_lists,
            lr_ratios,
            foreach=group["foreach"],
            nesterov=group["nesterov"],
            lr=group["lr"],
            weight_decay=group["wd"],
            cautious_wd=group["use_cautious_wd"],
            momentum=group["momentum"],
            ns_steps=group["ns_steps"],
            orthogonalization_strategy=group["orthogonalization_strategy"],
            maximize=group["maximize"],
            split_sizes=group["split_sizes"],
            **{key: group[key] for key in algorithm.options},
        )

    def _step_adam_group(self, group: dict) -> None:
        """Process a single Adam/AdamW param group."""
        params_with_grad: list[Tensor] = []
        grads: list[Tensor] = []
        state_steps: list[Tensor] = []
        exp_avgs: list[Tensor] = []
        exp_avg_sqs: list[Tensor] = []
        max_exp_avg_sqs: list[Tensor] = []
        beta1, beta2 = group["betas"]
        capturable = torch.compiler.is_compiling()

        self._init_adam_group(
            group,
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            capturable,
        )

        lr = group["lr"]
        weight_decay = group["wd"]
        if not group["fused"] and not torch.compiler.is_compiling():
            lr = float(lr) if torch.is_tensor(lr) else lr
            weight_decay = (
                float(weight_decay) if torch.is_tensor(weight_decay) else weight_decay
            )

        adam(
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            amsgrad=group["amsgrad"],
            has_complex=group["has_complex"],
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=group["eps"],
            maximize=group["maximize"],
            capturable=capturable,
            differentiable=False,
            fused=group["fused"],
            foreach=group["foreach"],
            grad_scale=getattr(self, "grad_scale", None),
            found_inf=getattr(self, "found_inf", None),
            decoupled_weight_decay=group["decoupled_weight_decay"],
        )

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Parameter groups with ``use_muon=True`` use their registered Muon-family
        algorithm; the remaining groups use PyTorch's functional Adam/AdamW
        implementation.

        Args:
            closure (Callable, optional): A closure that reevaluates the model and
                returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon_group(group)
            else:
                self._step_adam_group(group)

        return loss
