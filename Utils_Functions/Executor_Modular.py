"""
Modular Executor for SNN Training/Testing
==========================================

Architecture overview:

    ForwardPass          — which layers fire, returns (decode_out, spike_tensor)
    InputStrategy        — how to pick x_t at each timestep (teacher forcing / autoregression)
    LossStrategy         — which slice of predictions vs targets to compare
    TrainingHook(s)      — callbacks that run after each batch (spike monitor, NaN guard, ...)
    OptimizerScheduler   — when / how to swap optimizers between epochs
    DynamicsNetUpdater   — optional self-supervised update for f_phi
    Executor             — orchestrator: plugs all of the above together

Typical usage
-------------
    forward_pass      = ThreeLayerForward(module)
    input_strategy    = ScheduledTeacherForcing(percentage=1.0, interleave=True)
    loss_strategy     = FullSequenceLoss()
    hooks             = [SpikeMonitor(warn_below=1000, abort_below=1),
                         NaNGuard()]
    opt_scheduler     = EpochSwapScheduler(swap_every=5)

    executor = Executor(logger, module,
                        forward_pass=forward_pass,
                        input_strategy=input_strategy,
                        loss_strategy=loss_strategy,
                        hooks=hooks,
                        optimizer_scheduler=opt_scheduler)

    executor.train()
    executor.test()
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import numpy as np

import torch
import torch.nn as nn

from Utils_Functions.Utils import interleave_pos_neg_v2

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _maybe_interleave(x: torch.Tensor, do_interleave: bool) -> torch.Tensor:
    return interleave_pos_neg_v2(x) if do_interleave else x


def send_abort_email(spike_count, sender_email, sender_password, recipient_email):
    subject = "⚠️ Script Aborted: No Spikes Detected"
    body = f"""
    The script was aborted due to a critical error.

    Reason: No spikes detected
    Spike count recorded: {spike_count}

    Please investigate the issue.
    """

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        print("Abort notification email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Forward Passes
# ──────────────────────────────────────────────────────────────────────────────

class ForwardPass(ABC):
    """
    Encapsulates a single forward step through the network.

    Returns
    -------
    decode_out   : Tensor  — the decoded output at timestep t
    spike_tensor : Tensor  — tensor whose .sum() gives the spike count to monitor
    """

    @abstractmethod
    def reset(self, module) -> None:
        """Reset all relevant layer states."""

    @abstractmethod
    def detach(self, module) -> None:
        """Detach all layer states (TBPTT)."""

    @abstractmethod
    def __call__(self, module, x_t: torch.Tensor):
        ...


class TwoLayerForward(ForwardPass):
    """Recurrent → Decode  (layer1 = recurrent, layer2 = decode)."""

    def reset(self, module):
        module.layer1.reset()
        module.layer2.reset()

    def detach(self, module):
        module.layer1.detach_state()
        module.layer2.detach_state()

    def __call__(self, module, x_t):
        recurrent_out, _, _ = module.layer1.forward(x_t)
        decode_out = module.layer2.forward(recurrent_out.squeeze(-1))
        return decode_out, recurrent_out          # spike_tensor = recurrent_out


class ThreeLayerForward(ForwardPass):
    """Encode → Recurrent → Decode  (layer1 = enc, layer2 = rec, layer3 = dec)."""

    def reset(self, module):
        module.layer1.reset()
        module.layer2.reset()
        module.layer3.reset()

    def detach(self, module):
        module.layer1.detach_state()
        module.layer2.detach_state()
        module.layer3.detach_state()

    def __call__(self, module, x_t):
        spikes_out, _          = module.layer1.forward(x_t)
        recurrent_out, _, _    = module.layer2.forward(spikes_out)
        decode_out             = module.layer3.forward(recurrent_out.squeeze(-1))
        return decode_out, recurrent_out          # spike_tensor = recurrent_out


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Input Strategies  (how to pick x_t each timestep)
# ──────────────────────────────────────────────────────────────────────────────

class InputStrategy(ABC):
    """
    Decides what x_t to feed at the *next* timestep.

    Parameters available at call time
    ----------------------------------
    t            : current timestep index
    T            : total timesteps
    x            : full input batch  [B, T, F]
    decode_out   : network output at timestep t  [B, F]
    interleave   : whether to apply pos/neg interleaving
    percentage   : current teacher-forcing percentage  (may change during training)
    """

    @abstractmethod
    def __call__(self,
                 t: int, T: int,
                 x: torch.Tensor,
                 decode_out: torch.Tensor,
                 interleave: bool,
                 percentage: float) -> torch.Tensor:
        ...


class TeacherForcingInput(InputStrategy):
    """
    Classic teacher forcing: feed ground truth for the first `percentage` of T,
    then switch to autoregression.
    """

    def __call__(self, t, T, x, decode_out, interleave, percentage):
        if t < int(percentage * T):
            return _maybe_interleave(x[:, t + 1], interleave)
        return _maybe_interleave(decode_out.detach(), interleave)


class FullAutoregressiveInput(InputStrategy):
    """Always autoregress — no teacher forcing."""

    def __call__(self, t, T, x, decode_out, interleave, percentage):
        return _maybe_interleave(decode_out.detach(), interleave)


class FullTeacherInput(InputStrategy):
    """Always use ground truth — useful for debugging or first-phase training."""

    def __call__(self, t, T, x, decode_out, interleave, percentage):
        return _maybe_interleave(x[:, t + 1], interleave)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Loss Strategies  (which slice of predictions vs targets)
# ──────────────────────────────────────────────────────────────────────────────

class LossStrategy(ABC):
    """
    Selects the relevant slice of `predictions` and `targets` and calls
    `loss_fn`.
    """

    @abstractmethod
    def __call__(self,
                 loss_fn: Callable,
                 predictions: torch.Tensor,   # [B, T, F]
                 targets: torch.Tensor,        # [B, T, F]
                 t: int,
                 T: int,
                 percentage: float) -> torch.Tensor:
        ...


class FullSequenceLoss(LossStrategy):
    """Loss over the entire predicted sequence vs targets."""

    def __call__(self, loss_fn, predictions, targets, t, T, percentage):
        return loss_fn(predictions, targets)


class LastStepLoss(LossStrategy):
    """Loss only on the final timestep — pseudo-RL / goal-conditioned style."""

    def __call__(self, loss_fn, predictions, targets, t, T, percentage):
        return loss_fn(predictions[:, -1, :], targets[:, -1, :])


class PartialSequenceLoss(LossStrategy):
    """
    Loss only on the autoregressive portion of the sequence
    (i.e., timesteps after teacher forcing ended).
    """

    def __call__(self, loss_fn, predictions, targets, t, T, percentage):
        start = int(percentage * T)
        return loss_fn(predictions[:, start:, :], targets[:, start:, :])


class OffsetSequenceLoss(LossStrategy):
    """
    Loss over the full sequence but with a 1-step target offset.
    Useful when predictions[t] should match targets[t+1].
    """

    def __call__(self, loss_fn, predictions, targets, t, T, percentage):
        return loss_fn(predictions[:, :t + 1, :], targets[:, 1:, :])


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Training Hooks  (run after each batch — can abort training)
# ──────────────────────────────────────────────────────────────────────────────

class TrainingHook(ABC):
    """
    Called at the end of each batch.
    Return False to signal that training should abort.
    """

    @abstractmethod
    def on_batch_end(self, *, spike_count: float, module, logger, attempt: int,
                     **kwargs) -> bool:
        ...


class SpikeMonitor(TrainingHook):
    """
    Warns when spike count is below `warn_below`.
    Aborts (returns False) when spike count is below `abort_below`.
    Optionally resets the learning rate on abort so the outer retry loop can
    restart training.
    """

    def __init__(self, warn_below: int = 1000, abort_below: int = 1,
                 lr_reset_fn: Optional[Callable] = None):
        self.warn_below = warn_below
        self.abort_below = abort_below
        self.lr_reset_fn = lr_reset_fn          # e.g. lambda attempt, module: ...

    def on_batch_end(self, *, spike_count, module, logger, attempt, **kwargs):
        if spike_count < self.warn_below:
            logger.logger.warning(f"Low spike count: {spike_count}")
        if spike_count < self.abort_below:
            logger.logger.warning(
                f"No spikes detected: {spike_count} → "
                f"Retrying (attempt {attempt + 1})"
            )
            send_abort_email(
                spike_count=0,
                sender_email="miriam.barborini@gmail.com",
                sender_password="fdwy vxdw hhka pwhp",  # Use an App Password, not your real password
                recipient_email="miriam.barborini@gmail.com"
            )
            if self.lr_reset_fn is not None:
                self.lr_reset_fn(attempt, module)
            return False
        return True


class NaNGuard(TrainingHook):
    """Aborts immediately if any optimised parameter contains NaN."""

    def on_batch_end(self, *, module, logger, **kwargs):
        if any(torch.isnan(t).any() for t in module.opt_list):
            logger.logger.warning("NaN values detected → Aborting.")
            send_abort_email(
                spike_count=0,
                sender_email="miriam.barborini@gmail.com",
                sender_password="fdwy vxdw hhka pwhp",  # Use an App Password, not your real password
                recipient_email="miriam.barborini@gmail.com"
            )
            sys.exit()
        return True


class LossThresholdSwitch(TrainingHook):
    """
    Flips `teaching_percentage` to `new_percentage` once the rolling average
    of the last `window` losses drops below `threshold`.

    The hook writes to a shared mutable dict so the Executor can read it.
    """

    def __init__(self, threshold: float = 5.0, window: int = 10,
                 new_percentage: float = 0.5, state: Optional[dict] = None):
        self.threshold = threshold
        self.window = window
        self.new_percentage = new_percentage
        self.state = state if state is not None else {}     # shared mutable dict
        self.state.setdefault("switched", False)
        self.state.setdefault("losses", [])

    def on_batch_end(self, **kwargs):
        return True     # logic runs at epoch end — see on_epoch_end

    def on_epoch_end(self, *, epoch_loss: float, logger, **kwargs) -> dict:
        self.state["losses"].append(epoch_loss)
        losses = self.state["losses"]
        if (
            not self.state["switched"]
            and len(losses) > self.window
            and all(l < self.threshold for l in losses[-self.window:])
        ):
            logger.logger.info(
                f"Loss below {self.threshold} for {self.window} epochs → "
                f"switching teaching percentage to {self.new_percentage}"
            )
            self.state["switched"] = True
            self.state["new_percentage"] = self.new_percentage
        return self.state


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Optimizer Schedulers  (when / how to swap optimizers between epochs)
# ──────────────────────────────────────────────────────────────────────────────

class OptimizerScheduler(ABC):

    @abstractmethod
    def on_epoch_start(self, epoch: int, module, logger) -> None:
        ...


class StaticOptimizer(OptimizerScheduler):
    """Never changes the optimizer."""

    def on_epoch_start(self, epoch, module, logger):
        pass


class EpochSwapScheduler(OptimizerScheduler):
    """
    Alternates between two optimizer dictionaries every `swap_every` epochs.
    Requires `module.optimizer_dictionaries` to be a list of two dicts and
    `module.set_optimizer(dict)` to be defined.
    """

    def __init__(self, swap_every: int):
        self.swap_every = swap_every
        self._idx = 0

    def on_epoch_start(self, epoch, module, logger):
        if epoch > 0 and epoch % self.swap_every == 0:
            self._idx = 1 - self._idx
            module.set_optimizer(module.optimizer_dictionaries[self._idx])
            logger.logger.info(
                f"Epoch {epoch}: swapped to optimizer index {self._idx} "
                f"({module.opt_list_names_temp})"
            )
        elif epoch == 0:
            module.set_optimizer(module.optimizer_dictionaries[self._idx])
            logger.logger.info(f"Initial optimizer: {module.opt_list_names_temp}")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Dynamics Net Updater  (optional self-supervised f_phi step)
# ──────────────────────────────────────────────────────────────────────────────

class DynamicsNetUpdater:
    """
    Wraps the self-supervised update of the dynamics network.
    Pass an instance to Executor; set to None to skip.

    Parameters
    ----------
    pos_slice : slice or tuple — indices for position channels in decode_out
    vel_slice : slice or tuple — indices for velocity channels in decode_out
    only_after_switch : bool  — if True, only update after SWITCH becomes True
    """

    def __init__(self, pos_slice=slice(0, 2), vel_slice=slice(2, 4),
                 only_after_switch: bool = False):
        self.pos_slice = pos_slice
        self.vel_slice = vel_slice
        self.only_after_switch = only_after_switch

    def update(self, module, decode_layer_full: torch.Tensor,
               switched: bool) -> Optional[float]:
        if module.dynamics_net is None:
            return None
        if self.only_after_switch and not switched:
            return None
        f_loss = module.dynamics_net.self_supervised_update(
            decode_layer_full[..., self.pos_slice],
            decode_layer_full[..., self.vel_slice],
        )
        return f_loss


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Executor  (the orchestrator)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutorConfig:
    """Convenience dataclass so callers can pass one object instead of many kwargs."""

    forward_pass:           ForwardPass
    input_strategy:         InputStrategy        = field(default_factory=TeacherForcingInput)
    loss_strategy:          LossStrategy         = field(default_factory=FullSequenceLoss)
    hooks:                  List[TrainingHook]   = field(default_factory=list)
    optimizer_scheduler:    OptimizerScheduler   = field(default_factory=StaticOptimizer)
    dynamics_updater:       Optional[DynamicsNetUpdater] = None
    interleave:             bool                 = True
    initial_teaching_pct:   float                = 1.0      # overridden at test time
    test_teaching_pct:      float                = 0.5
    max_retries:            int                  = 3
    inner_runtime:          Optional[int]        = None     # overrides module.tp


class Executor:
    """
    Plug-and-play training / testing executor.

    All behaviour is determined by the components injected via ExecutorConfig.
    """

    def __init__(self, logger, module, config: ExecutorConfig):
        self.logger  = logger
        self.module  = module
        self.cfg     = config

        self.epochs          = module.tp["epochs"]
        self.inner_runtime   = (config.inner_runtime
                                or module.tp["network_small_runtime"])
        self._loss_switch_hook: Optional[LossThresholdSwitch] = next(
            (h for h in config.hooks if isinstance(h, LossThresholdSwitch)), None
        )

    # ── public API ──────────────────────────────────────────────────────────

    def train(self, plot: bool = True) -> Optional[list]:
        return self._run(actually_train=True, plot=plot)

    def test(self, plot: bool = True) -> Optional[list]:
        return self._run(actually_train=False, plot=plot)

    def plot(self):
        self.logger.save_figures(self.module)

    # ── internals ───────────────────────────────────────────────────────────

    def _run(self, actually_train: bool, plot: bool) -> Optional[list]:
        for attempt in range(self.cfg.max_retries):
            result = self._attempt(actually_train, plot, attempt)
            if result is not None:
                return result
            if not actually_train:
                return None                 # no retries for test mode
        self.logger.logger.warning("Aborted after maximum retries.")
        send_abort_email(
            spike_count=0,
            sender_email="miriam.barborini@gmail.com",
            sender_password="fdwy vxdw hhka pwhp",  # Use an App Password, not your real password
            recipient_email="miriam.barborini@gmail.com"
        )
        return None

    def _attempt(self, actually_train: bool, plot: bool,
                 attempt: int) -> Optional[list]:

        # ── data source ─────────────────────────────────────────────────────
        if actually_train:
            name, data = "train", self.module.train_dataloader
        else:
            name, data = "test",  self.module.test_dataloader
            if data is None:
                return False

        tot_epochs = self.epochs + 1 if actually_train else 1 # add the 1 so that it is a round number and it gets saved

        # ── mutable state shared across epochs ──────────────────────────────
        teaching_pct = (self.cfg.initial_teaching_pct if actually_train
                        else self.cfg.test_teaching_pct)
        switched     = False

        losses, outputs, targets, spikes_rasters = [], [], [], []

        self.logger.logger.info(f"Teaching percentage: {teaching_pct:.2f}")

        for epoch in range(tot_epochs):

            # ── optimizer scheduling ─────────────────────────────────────────
            self.cfg.optimizer_scheduler.on_epoch_start(epoch, self.module, self.logger)

            if name == "train":
                self.logger.logger.info(f"Optimizer: {self.module.opt_list_names}")

            total_loss, spike_count = 0.0, 0

            # ── batch loop ───────────────────────────────────────────────────
            abort = False
            for x, y in data:
                batch_result = self._run_batch(
                    x, y, actually_train, teaching_pct, switched, attempt
                )
                if batch_result is None:           # hook requested abort/retry
                    abort = True
                    break
                loss_val, spike_count_batch, decode_layer_full, y_cpu, spikes_raster_batch = batch_result
                total_loss  += loss_val
                spike_count += spike_count_batch
                
            if abort:
                return None                        # trigger retry

            # ── epoch bookkeeping ────────────────────────────────────────────
            epoch_loss = total_loss / len(data)
            losses.append(epoch_loss)

            if epoch == 0 or epoch == tot_epochs - 1:
                outputs.append(decode_layer_full.detach().cpu().numpy())
                targets.append(y_cpu)
                spikes_rasters.append(spikes_raster_batch)

            self.logger.logger.info(
                f"Epoch {epoch + 1}/{tot_epochs}: "
                f"Loss = {epoch_loss:.6f}  "
                f"Spikes = {spike_count}"
            )

            # ── adaptive teaching switch (via LossThresholdSwitch hook) ──────
            if self._loss_switch_hook is not None and name == "train":
                state = self._loss_switch_hook.on_epoch_end(
                    epoch_loss=epoch_loss, logger=self.logger
                )
                if state.get("switched") and not switched:
                    teaching_pct = state.get(
                        "new_percentage",
                        self.module.tp.get("runtime_percentage_teaching", 0.5)
                    )
                    switched = True

        # ── save & return ────────────────────────────────────────────────────
        results = {"losses": losses, "output": outputs, "target": targets}
        self.logger.save_results(self.module, results, name=f"{name}_results", spikes_rasters=spikes_rasters)
        
        if actually_train and plot:
            self.logger.save_model(self.module)
            self.logger.save_figures(self.module, name=name)
        elif plot:
            self.logger.save_figures(self.module, name=name)

        return outputs

    def _run_batch(self, x, y, actually_train, teaching_pct,
                   switched, attempt):
        """
        Execute one full batch: forward through time, compute loss,
        optionally back-prop.

        Returns (loss_value, spike_count, decode_layer_full, y_cpu)
        or None to signal abort/retry.
        """
        cfg    = self.cfg
        module = self.module
        logger = self.logger

        # ── setup ─────────────────────────────────────────────────────────
        cfg.forward_pass.reset(module)
        x, y = x.to(module.device), y.to(module.device)
        B, T, _ = x.shape

        x_t = _maybe_interleave(x[:, 0], cfg.interleave)

        decode_layer_full = []
        spikes_raster = []
        spike_count       = 0

        # ── temporal loop ─────────────────────────────────────────────────
        for t in range(T):

            for _ in range(self.inner_runtime):
                decode_out, spike_tensor = cfg.forward_pass(module, x_t)
                spikes_raster.append(spike_tensor.detach().cpu())
                spike_count += torch.sum(spike_tensor).item()

            # next input
            x_t = cfg.input_strategy(
                t, T, x, decode_out, cfg.interleave, teaching_pct
            )

            decode_layer_full.append(decode_out)
            cfg.forward_pass.detach(module)

        decode_layer_full = torch.stack(decode_layer_full, dim=1)  # [B, T-1, F]

        # ── dynamics net (optional) ────────────────────────────────────────
        if actually_train and cfg.dynamics_updater is not None:
            cfg.dynamics_updater.update(module, decode_layer_full, switched)

        # ── loss ──────────────────────────────────────────────────────────
        loss = cfg.loss_strategy(
            module.loss_fn, decode_layer_full, y, t, T, teaching_pct
        )
        
        # ── dynamics net loss adaptation (optional) ────────────────────────────────────────       
        loss_metrics = None
        if isinstance(loss, tuple):
            loss, loss_metrics = loss

        # ── back-prop ─────────────────────────────────────────────────────
        if actually_train:
            module.optimizer.zero_grad()
            loss.backward()
            module.optimizer.step()
            if loss_metrics is not None:
                logger.logger.debug(
                    "Loss breakdown — " +
                    "  ".join(f"{k}: {v:.4f}" for k, v in loss_metrics.items())
                )

        # ── hooks ─────────────────────────────────────────────────────────
        if actually_train:
            for hook in cfg.hooks:
                ok = hook.on_batch_end(
                    spike_count=spike_count,
                    module=module,
                    logger=logger,
                    attempt=attempt,
                )
                if not ok:
                    return None

        return loss.item(), spike_count, decode_layer_full, y.detach().cpu().numpy(), spikes_raster


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Pre-built Configurations  (drop-in replacements for your old methods)
# ──────────────────────────────────────────────────────────────────────────────

def make_executor_test1(logger, module, interleave=True) -> Executor:
    """Equivalent to the old Executor.test_1 / train_1."""
    cfg = ExecutorConfig(
        forward_pass     = TwoLayerForward(),
        input_strategy   = TeacherForcingInput(),
        loss_strategy    = FullSequenceLoss(),
        hooks            = [SpikeMonitor(warn_below=1000, abort_below=1), NaNGuard()],
        optimizer_scheduler = StaticOptimizer(),
        interleave       = interleave,
        initial_teaching_pct = module.tp.get("runtime_percentage_teaching", 1.0),
    )
    return Executor(logger, module, cfg)


def make_executor_test3(logger, module, epoch_swap: int,
                        interleave=True) -> Executor:
    """Equivalent to the old Executor.test_3 / train_3."""
    switch_state = {}
    switch_hook  = LossThresholdSwitch(
        threshold=5.0, window=10,
        new_percentage=module.tp.get("runtime_percentage_teaching", 0.5),
        state=switch_state,
    )
    cfg = ExecutorConfig(
        forward_pass        = ThreeLayerForward(),
        input_strategy      = TeacherForcingInput(),
        loss_strategy       = OffsetSequenceLoss(),
        hooks               = [switch_hook,
                                SpikeMonitor(warn_below=1000, abort_below=1),
                                NaNGuard()],
        optimizer_scheduler = EpochSwapScheduler(swap_every=epoch_swap),
        dynamics_updater    = DynamicsNetUpdater(only_after_switch=True),
        interleave          = interleave,
        initial_teaching_pct = 1.0,
        test_teaching_pct    = 0.5,
    )
    return Executor(logger, module, cfg)


def make_executor_pseudoRL(logger, module, interleave=True) -> Executor:
    """Equivalent to the old test_pseudoRL / train_pseudoRL."""
    cfg = ExecutorConfig(
        forward_pass        = ThreeLayerForward(),
        input_strategy      = FullAutoregressiveInput(),
        loss_strategy       = LastStepLoss(),
        hooks               = [SpikeMonitor(warn_below=1000, abort_below=1), NaNGuard()],
        optimizer_scheduler = StaticOptimizer(),
        dynamics_updater    = DynamicsNetUpdater(only_after_switch=True),
        interleave          = interleave,
        initial_teaching_pct = 1.0,
        test_teaching_pct    = 0.5,
    )
    return Executor(logger, module, cfg)


def make_executor_pseudoRL_fullloss(logger, module, interleave=True) -> Executor:
    """Equivalent to test_pseudoRL_fullloss / train_pseudoRL_fullloss."""
    cfg = ExecutorConfig(
        forward_pass        = ThreeLayerForward(),
        input_strategy      = FullAutoregressiveInput(),
        loss_strategy       = FullSequenceLoss(),
        hooks               = [SpikeMonitor(warn_below=1000, abort_below=1), NaNGuard()],
        optimizer_scheduler = StaticOptimizer(),
        dynamics_updater    = DynamicsNetUpdater(only_after_switch=True),
        interleave          = interleave,
        initial_teaching_pct = 1.0,
        test_teaching_pct    = 0.5,
    )
    return Executor(logger, module, cfg)


def make_executor_report1(logger, module, interleave=True) -> Executor:
    """Equivalent to test_report_1 / train_report_1."""
    cfg = ExecutorConfig(
        forward_pass        = ThreeLayerForward(),
        input_strategy      = TeacherForcingInput(),
        loss_strategy       = FullSequenceLoss(),
        hooks               = [SpikeMonitor(warn_below=1000, abort_below=1), NaNGuard()],
        optimizer_scheduler = StaticOptimizer(),
        interleave          = interleave,
        initial_teaching_pct = module.tp.get("runtime_percentage_teaching", 1.0),
    )
    return Executor(logger, module, cfg)

def make_executor_report6(logger, module, interleave=True) -> Executor:
    cfg = ExecutorConfig(
        forward_pass        = ThreeLayerForward(),
        input_strategy      = TeacherForcingInput(),
        loss_strategy       = FullSequenceLoss(),
        hooks               = [SpikeMonitor(warn_below=1000, abort_below=1), NaNGuard()],
        optimizer_scheduler = StaticOptimizer(),
        dynamics_updater    = DynamicsNetUpdater(only_after_switch=False),
        interleave          = interleave,
        initial_teaching_pct = module.tp.get("runtime_percentage_teaching", 0.5),
        test_teaching_pct    = module.tp.get("test_runtime_percentage_teaching", 0.5),
    )
    return Executor(logger, module, cfg)


    