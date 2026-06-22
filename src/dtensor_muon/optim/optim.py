import math
from typing import cast

import torch
from torch import Tensor
from torch.optim.adam import adam
from torch.optim.optimizer import _device_dtype_check_for_fused, _get_scalar_dtype

from dtensor_muon.orthogonalize import OrthogonalizationStrategy, zeropower


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        wd: float = 0.1,
        use_cautious_wd: bool = True,
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
        fused_adam: bool = True,
        maximize: bool = False,
        compile: bool = False,
    ):
        """
        Unified Muon + Adam optimizer.

        Args:
            params: An iterable of parameters or an iterable of param-group dicts.
                Each param-group dict can have an optional "algorithm" key:
                - "adamw" or "adam": Use Adam/AdamW for this group
                - "muon" or omitted: Use Muon for this group (default)
            lr: Default learning rate
            wd: Default weight decay
            use_cautious_wd: Use cautious weight decay for Muon groups. When enabled,
                weight decay is only applied when the update and parameter have the
                same sign (i.e., when u * p > 0).
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
            compile: Compile per-parameter optimizer step
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
        self.maximize = maximize
        self.compile = compile

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

            if algorithm in ("adamw", "adam"):
                param_groups.append(self._build_adam_group(group))
            elif algorithm == "muon":
                param_groups.append(self._build_muon_group(group))
            else:
                raise ValueError(
                    f"Unknown algorithm '{algorithm}'. Must be 'muon', 'adam', or 'adamw'."
                )

        super().__init__(param_groups, {})

        self._init_step_impls()

    def _init_step_impls(self) -> None:
        if self.compile:
            self._adam_impl = torch.compile(adam, dynamic=True)
            self._muon_impl = torch.compile(self.muon, dynamic=True)
        else:
            self._adam_impl = adam
            self._muon_impl = self.muon

    def _build_muon_group(self, group: dict):
        if not all(p.ndim >= 2 for p in group["params"]):
            raise ValueError(
                "Muon only supports 2D+ parameters; found a 1D tensor in a Muon group"
            )

        if any(torch.is_complex(p) for p in group["params"]):
            raise NotImplementedError(
                "Complex parameters are not supported in Muon. Add these parameters to the Adam group or use a different optimizer."
            )

        return {
            "params": group["params"],
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
            "flatten": group.get("flatten", True),
            # Set to true for foreach zero_grad
            "foreach": group.get("foreach", True),
        }

    def _build_adam_group(self, group: dict):
        return {
            "params": group["params"],
            "use_muon": False,
            "lr": torch.tensor(group.get("lr", self.lr)),
            "wd": torch.tensor(group.get("wd", self.wd)),
            "amsgrad": group.get("amsgrad", self.amsgrad),
            "betas": group.get("betas", self.adam_betas),
            "eps": group.get("eps", self.adam_eps),
            "decoupled_weight_decay": group.get("decoupled_weight_decay", self.is_adamw),
            "fused": group.get("fused", self.fused_adam),
            "maximize": group.get("maximize", self.maximize),
            "has_complex": any(torch.is_complex(p) for p in group["params"]),
        }

    def __setstate__(self, state):
        super().__setstate__(state)
        self.compile = getattr(self, "compile", False)

        for group in self.param_groups:
            # Muon state initialization
            if group["use_muon"]:
                group.setdefault("ns_steps", 5)
                group.setdefault("nesterov", True)
                group.setdefault("flatten", True)
                group.setdefault("use_cautious_wd", True)
                group.setdefault("orthogonalization_strategy", "newton_schulz")
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

        self._init_step_impls()

    def _init_muon_group(
        self,
        group,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        momentum_buffers: list[Tensor],
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
                assert grad.ndim == 3 or group["flatten"], (
                    f"Got ndim={grad.ndim}. Please set flatten=True"
                )
                grad = grad.view(grad.size(0), -1) if group["flatten"] else grad

            grads.append(grad)
            state = self.state[p]

            # Lazy state initialization
            if len(state) == 0:
                state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                state["momentum_buffer"] = torch.zeros_like(
                    grad, memory_format=torch.preserve_format
                )
                state["lr_ratio"] = torch.tensor(
                    math.sqrt(max(1.0, grad.shape[-2] / grad.shape[-1]))
                )

            state["step"] += 1
            momentum_buffers.append(state["momentum_buffer"])
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
                    torch.zeros((), dtype=_get_scalar_dtype(is_fused=True), device=p.device)
                    if group["fused"]
                    else torch.tensor(0.0, dtype=_get_scalar_dtype())
                )

                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if group["amsgrad"]:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            if group["amsgrad"]:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])

            state_steps.append(state["step"])

    def muon(
        self,
        params: list[Tensor],
        grads: list[Tensor],
        momentum_buffers: list[Tensor],
        lr_ratios: list[Tensor],
        *,
        nesterov: bool,
        lr: Tensor,
        weight_decay: float,
        cautious_wd: bool,
        momentum: float,
        ns_steps: int,
        orthogonalization_strategy: OrthogonalizationStrategy,
        maximize: bool,
    ) -> None:
        """
        Muon update for a list of parameters. Override in subclasses to customize.

        Handles device/dtype grouping, shape grouping, batching, and calls
        the underlying foreach implementation.
        """
        if len(params) == 0:
            return

        for param, grad, momentum_buffer, lr_ratio in zip(
            params, grads, momentum_buffers, lr_ratios, strict=True
        ):
            if maximize:
                grad.neg_()

            param_fp32 = param.to(torch.float32)
            grad_fp32 = grad.to(torch.float32)
            momentum_buffer_fp32 = momentum_buffer.to(torch.float32)

            # Update momentum buffer (in fp32)
            momentum_buffer_fp32.mul_(momentum).add_(grad_fp32)
            momentum_buffer.copy_(momentum_buffer_fp32)

            # Update gradient
            if nesterov:
                grad_fp32.add_(momentum_buffer_fp32, alpha=momentum)
            else:
                grad_fp32 = momentum_buffer_fp32

            # Zero power via orthogonalization (Newton-Schulz or Polar Express)
            u = zeropower(grad_fp32, steps=ns_steps, strategy=orthogonalization_strategy)
            u = u.view_as(param_fp32)

            # Apply weight decay
            if weight_decay != 0:
                if cautious_wd:
                    u.addcmul_(param_fp32, (u * param_fp32 > 0), value=weight_decay)
                else:
                    u.add_(param_fp32, alpha=weight_decay)

            # Scale update
            adjusted_lr = lr_ratio * lr
            param_fp32.add_(u, alpha=-cast(float, adjusted_lr))
            param.copy_(param_fp32)

    def _step_muon_group(self, group: dict) -> None:
        """Process a single Muon param group."""
        params_with_grad: list[Tensor] = []
        grads: list[Tensor] = []
        state_steps: list[Tensor] = []
        momentum_buffers: list[Tensor] = []
        lr_ratios: list[Tensor] = []

        self._init_muon_group(
            group,
            params_with_grad,
            grads,
            momentum_buffers,
            lr_ratios,
            state_steps,
        )

        self._muon_impl(
            params_with_grad,
            grads,
            momentum_buffers,
            lr_ratios,
            nesterov=group["nesterov"],
            lr=group["lr"],
            weight_decay=group["wd"],
            cautious_wd=group["use_cautious_wd"],
            momentum=group["momentum"],
            ns_steps=group["ns_steps"],
            orthogonalization_strategy=group["orthogonalization_strategy"],
            maximize=group["maximize"],
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

        self._init_adam_group(
            group,
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
        )

        self._adam_impl(
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
            lr=group["lr"],
            weight_decay=group["wd"],
            eps=group["eps"],
            maximize=group["maximize"],
            foreach=not group["fused"],
            capturable=False,
            differentiable=False,
            fused=group["fused"],
            grad_scale=getattr(self, "grad_scale", None),
            found_inf=getattr(self, "found_inf", None),
            decoupled_weight_decay=group["decoupled_weight_decay"],
        )

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Internally the method distinguishes between parameters that should use the
        Muon update (``state[p]["use_muon"] == True``) and those that should fall back
        to Adam/AdamW (``state[p]["use_muon"] == False``).  The latter path calls the
        fused implementation provided by ``torchao.optim.adam.single_param_adam``.

        Args:
            closure (Callable, optional): A closure that reevaluates the model and
                returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch._dynamo.utils.disable_cache_limit():
            muon_groups = [g for g in self.param_groups if g["use_muon"]]
            adam_groups = [g for g in self.param_groups if not g["use_muon"]]

            for group in muon_groups:
                self._step_muon_group(group)
            for group in adam_groups:
                self._step_adam_group(group)

        return loss
