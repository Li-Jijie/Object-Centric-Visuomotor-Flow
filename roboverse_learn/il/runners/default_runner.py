import copy
import datetime
import json
import os
import pathlib
import random
import time
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import hydra
import imageio.v2 as iio
import numpy as np
import torch
import tqdm
import wandb
from loguru import logger as log

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.render import RenderCfg
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot
from metasim.task.registry import get_task_class
from metasim.randomization import DomainRandomizationManager, DRConfig
from roboverse_learn.il.utils.ema_model import EMAModel
from roboverse_learn.il.runners.base_runner import BaseRunner
from roboverse_learn.il.utils.json_logger import JsonLogger
from roboverse_learn.il.utils.lr_scheduler import get_scheduler
from roboverse_learn.il.utils.pytorch_util import optimizer_to
from roboverse_learn.il.utils.visualization import plot_all_latent_visualizations

RANDOMIZATION_AVAILABLE = True


class MujocoStateRandomizationManager:
    """IsaacSim-free fallback DR manager for MuJoCo.

    This manager perturbs initial object/robot XY poses per-demo and keeps
    compatibility with the DomainRandomizationManager API used by eval runners.
    """

    def __init__(self, config):
        self.config = config
        self.init_states = []
        self.original_positions = {}
        self._demo_offsets = {}
        seed = getattr(config, "randomization_seed", None)
        self._rng = np.random.default_rng(seed)

    def _get_or_sample_offset(self, demo_idx: int):
        if demo_idx in self._demo_offsets:
            return self._demo_offsets[demo_idx]

        # Conservative defaults to avoid invalid starts while still adding DR pressure.
        level = int(getattr(self.config, "level", 1))
        if level <= 1:
            pos_range = 0.02
            yaw_deg = 8.0
        elif level == 2:
            pos_range = 0.03
            yaw_deg = 12.0
        else:
            pos_range = 0.04
            yaw_deg = 15.0

        dx = float(self._rng.uniform(-pos_range, pos_range))
        dy = float(self._rng.uniform(-pos_range, pos_range))
        yaw = float(np.deg2rad(self._rng.uniform(-yaw_deg, yaw_deg)))
        self._demo_offsets[demo_idx] = (dx, dy, yaw)
        return self._demo_offsets[demo_idx]

    def apply_randomization(self, demo_idx: int = 0, is_initial: bool = False):
        # Offset is lazily sampled here; applied in update_positions_to_table.
        _ = self._get_or_sample_offset(demo_idx)

    def update_positions_to_table(self, demo_idx: int, env_id: int = 0):
        del env_id
        if demo_idx >= len(self.init_states):
            return

        init_state = self.init_states[demo_idx]
        demo_key = f"demo_{demo_idx}"
        if demo_key not in self.original_positions:
            return

        demo_original_positions = self.original_positions[demo_key]
        if not demo_original_positions:
            return

        dx, dy, yaw = self._get_or_sample_offset(demo_idx)
        c, s = float(np.cos(yaw)), float(np.sin(yaw))

        all_x = [demo_original_positions[k]["x"] for k in demo_original_positions]
        all_y = [demo_original_positions[k]["y"] for k in demo_original_positions]
        cx = sum(all_x) / len(all_x)
        cy = sum(all_y) / len(all_y)

        def _transform_xy(x, y):
            rx, ry = x - cx, y - cy
            nx = c * rx - s * ry + cx + dx
            ny = s * rx + c * ry + cy + dy
            return nx, ny

        for obj_name, obj_state in init_state["objects"].items():
            key = f"obj_{obj_name}"
            if key not in demo_original_positions:
                continue
            orig = demo_original_positions[key]
            nx, ny = _transform_xy(orig["x"], orig["y"])
            obj_state["pos"] = torch.tensor(
                [nx, ny, orig["z"]],
                dtype=obj_state["pos"].dtype,
                device=obj_state["pos"].device,
            )

        for robot_name, robot_state in init_state["robots"].items():
            key = f"robot_{robot_name}"
            if key not in demo_original_positions:
                continue
            orig = demo_original_positions[key]
            nx, ny = _transform_xy(orig["x"], orig["y"])
            robot_state["pos"] = torch.tensor(
                [nx, ny, orig["z"]],
                dtype=robot_state["pos"].dtype,
                device=robot_state["pos"].device,
            )

        log.info(
            f"[MuJoCo DR] demo={demo_idx}, dx={dx:.3f}, dy={dy:.3f}, yaw_deg={np.rad2deg(yaw):.2f}"
        )

    def update_camera_look_at(self, env_id: int = 0):
        del env_id
        return

    def apply_camera_randomization(self):
        return


def ensure_clean_state(handler, expected_state=None):
    """Ensure environment is in clean initial state with intelligent validation."""
    prev_state = None
    stable_count = 0
    max_steps = 10
    min_steps = 2

    for step in range(max_steps):
        handler.simulate()
        current_state = handler.get_states()

        if step >= min_steps:
            if prev_state is not None:
                is_stable = True
                if hasattr(current_state, "objects") and hasattr(prev_state, "objects"):
                    for obj_name, obj_state in current_state.objects.items():
                        if obj_name in prev_state.objects:
                            curr_dof = getattr(obj_state, "dof_pos", None)
                            prev_dof = getattr(prev_state.objects[obj_name], "dof_pos", None)
                            if curr_dof is not None and prev_dof is not None:
                                if not torch.allclose(curr_dof, prev_dof, atol=1e-5):
                                    is_stable = False
                                    break

                if is_stable and expected_state is not None:
                    is_correct_state = _validate_state_correctness(current_state, expected_state)
                    if not is_correct_state:
                        log.debug(f"State stable but incorrect at step {step}, continuing simulation...")
                        stable_count = 0
                        is_stable = False

                if is_stable:
                    stable_count += 1
                    if stable_count >= 2:
                        break
                else:
                    stable_count = 0

            prev_state = current_state

    if expected_state is not None:
        final_state = handler.get_states()
        is_final_correct = _validate_state_correctness(final_state, expected_state)
        if not is_final_correct:
            log.warning(f"State validation failed after {max_steps} steps - reset may not have taken full effect")

    handler.get_states()


def _validate_state_correctness(current_state, expected_state):
    """Validate that current state matches expected initial state for critical objects."""
    if not hasattr(current_state, "objects") or not hasattr(expected_state, "objects"):
        return True

    critical_objects = []
    for obj_name, expected_obj in expected_state.objects.items():
        if hasattr(expected_obj, "dof_pos") and getattr(expected_obj, "dof_pos", None) is not None:
            critical_objects.append(obj_name)

    if not critical_objects:
        return True

    tolerance = 5e-3

    for obj_name in critical_objects:
        if obj_name not in current_state.objects:
            continue

        expected_obj = expected_state.objects[obj_name]
        current_obj = current_state.objects[obj_name]

        expected_dof = getattr(expected_obj, "dof_pos", None)
        current_dof = getattr(current_obj, "dof_pos", None)

        if expected_dof is not None and current_dof is not None:
            if not torch.allclose(current_dof, expected_dof, atol=tolerance):
                diff = torch.abs(current_dof - expected_dof).max().item()
                log.debug(f"DOF mismatch for {obj_name}: max diff = {diff:.6f} (tolerance = {tolerance})")
                return False

    return True


def _parse_csv_names(raw_names: str):
    raw_names = str(raw_names or "")
    names = [name.strip() for name in raw_names.split(",") if name.strip()]
    return set(names) if names else None


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _write_json(path: pathlib.Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def apply_mid_rollout_object_perturbation(env, args, num_envs: int):
    """Move task object(s) after rollout history has accumulated."""
    states = env.handler.get_states()
    perturbed_states = copy.deepcopy(states)
    obj_names = _parse_csv_names(getattr(args, "mid_obj_perturb_obj_names", ""))
    dx = float(getattr(args, "mid_obj_perturb_x", 0.10))
    dy = float(getattr(args, "mid_obj_perturb_y", 0.0))
    dz = float(getattr(args, "mid_obj_perturb_z", 0.0))
    zero_vel = bool(getattr(args, "mid_obj_perturb_zero_vel", True))

    applied = []
    for obj_name, obj_state in perturbed_states.objects.items():
        if obj_names is not None and obj_name not in obj_names:
            continue
        root_state = obj_state.root_state.clone()
        offset = torch.tensor([dx, dy, dz], dtype=root_state.dtype, device=root_state.device)
        root_state[:num_envs, :3] = root_state[:num_envs, :3] + offset
        if zero_vel and root_state.shape[-1] >= 13:
            root_state[:num_envs, 7:13] = 0
        obj_state.root_state = root_state
        if zero_vel and getattr(obj_state, "joint_vel", None) is not None:
            obj_state.joint_vel = torch.zeros_like(obj_state.joint_vel)
        applied.append(obj_name)

    if not applied:
        available = ", ".join(perturbed_states.objects.keys())
        log.warning(
            "[MidObjPerturb] No matching objects for names={} | available={}",
            sorted(obj_names) if obj_names else "ALL",
            available,
        )
        return env.handler.get_states()

    env_ids = list(range(num_envs))
    env.handler.set_states(states=perturbed_states, env_ids=env_ids)
    if hasattr(env.handler, "refresh_render"):
        env.handler.refresh_render()
    log.info(
        "[MidObjPerturb] step={} objects={} offset=({:+.3f}, {:+.3f}, {:+.3f}) zero_vel={}",
        int(getattr(args, "mid_obj_perturb_step", 40)),
        ",".join(applied),
        dx,
        dy,
        dz,
        zero_vel,
    )
    return env.handler.get_states()


class DefaultRunner(BaseRunner):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.train_config.training_params.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model = hydra.utils.instantiate(cfg.policy_config)
        self.policy_name = cfg.policy_name

        self.ema_model = None
        if cfg.train_config.training_params.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.train_config.optimizer, params=self.model.parameters()
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

        self.eval_args = hydra.utils.instantiate(cfg.eval_config.eval_args)

    def train(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.train_config.training_params.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset = hydra.utils.instantiate(cfg.dataset_config)
        train_dataloader = create_dataloader(dataset, **cfg.train_config.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = create_dataloader(
            val_dataset, **cfg.train_config.val_dataloader
        )

        self.model.set_normalizer(normalizer)
        if cfg.train_config.training_params.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.train_config.training_params.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.train_config.training_params.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.train_config.training_params.num_epochs
            )
            // cfg.train_config.training_params.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.train_config.training_params.use_ema:
            ema = hydra.utils.instantiate(cfg.train_config.ema, model=self.ema_model)

        wandb_run = None

        # configure logging
        if cfg.logging.mode == "online":
            # Truncate tags to max 64 characters (wandb limit)
            logging_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            if "tags" in logging_cfg and logging_cfg["tags"]:
                logging_cfg["tags"] = [tag[:64] if len(tag) > 64 else tag for tag in logging_cfg["tags"]]
            
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **logging_cfg,
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # device transfer
        device = torch.device(cfg.train_config.training_params.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.train_config.training_params.debug:
            cfg.train_config.training_params.num_epochs = 2
            cfg.train_config.training_params.max_train_steps = 3
            cfg.train_config.training_params.max_val_steps = 3
            cfg.train_config.training_params.rollout_every = 1
            cfg.train_config.training_params.checkpoint_every = 1
            # Allow il_run.sh --val_every to disable validation
            pass  # val_every set from config
            cfg.train_config.training_params.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.train_config.training_params.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.train_config.training_params.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.train_config.training_params.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dataset.postprocess(batch, device)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss = self.model.compute_loss(batch)
                        loss = (
                            raw_loss
                            / cfg.train_config.training_params.gradient_accumulate_every
                        )
                        loss.backward()

                        # step optimizer (true gradient accumulation by batch index)
                        accumulate_every = max(
                            1, int(cfg.train_config.training_params.gradient_accumulate_every)
                        )
                        should_step = ((batch_idx + 1) % accumulate_every == 0) or (
                            batch_idx == (len(train_dataloader) - 1)
                        )
                        if should_step:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.train_config.training_params.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }
                        if hasattr(self.model, "get_last_loss_metrics"):
                            try:
                                extra_metrics = self.model.get_last_loss_metrics()
                                for k, v in extra_metrics.items():
                                    step_log[f"train/{k}"] = float(v)
                            except Exception:
                                pass

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if wandb_run is not None:
                                wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (
                            cfg.train_config.training_params.max_train_steps is not None
                        ) and batch_idx >= (
                            cfg.train_config.training_params.max_train_steps - 1
                        ):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.train_config.training_params.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.train_config.training_params.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                if cfg.train_config.training_params.val_every < 10000 and (self.epoch % cfg.train_config.training_params.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.train_config.training_params.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch, device)
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (
                                    cfg.train_config.training_params.max_val_steps
                                    is not None
                                ) and batch_idx >= (
                                    cfg.train_config.training_params.max_val_steps - 1
                                ):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # Latent space visualization for A2A policy
                if hasattr(policy, 'get_latents_for_visualization'):
                    try:
                        with torch.no_grad():
                            # Collect latents from multiple validation batches
                            all_history_latents = []
                            all_future_latents = []
                            max_samples = 500  # Limit samples for t-SNE performance
                            first_batch = None
                            
                            for batch_idx, batch in enumerate(val_dataloader):
                                batch = dataset.postprocess(batch, device)
                                if first_batch is None:
                                    first_batch = batch  # Save for trajectory visualization
                                history_latents, future_latents = policy.get_latents_for_visualization(batch)
                                all_history_latents.append(history_latents.cpu())
                                all_future_latents.append(future_latents.cpu())
                                
                                if sum(h.shape[0] for h in all_history_latents) >= max_samples:
                                    break
                            
                            # Concatenate all collected latents
                            history_latents = torch.cat(all_history_latents, dim=0)[:max_samples]
                            future_latents = torch.cat(all_future_latents, dim=0)[:max_samples]
                            
                            # Get flow trajectories for visualization (uses model's num_sampling_steps)
                            trajectories = None
                            trajectory_targets = None
                            if hasattr(policy, 'get_flow_trajectories') and first_batch is not None:
                                policy.output_dir = str(self.output_dir)
                                policy.eval()  # disable Dropout for clean trajectories
                                raw = policy.get_flow_trajectories(first_batch, n_samples=5)
                                if isinstance(raw[0], tuple) and len(raw[0]) == 2:
                                    (trajectories, trajectories_base), trajectory_targets = raw
                                else:
                                    trajectories, trajectory_targets = raw
                                    trajectories_base = None
                                policy.train()  # restore training mode

                            # Generate all visualizations
                            viz_dir = pathlib.Path(self.output_dir) / "latent_viz"
                            viz_results = plot_all_latent_visualizations(
                                history_latents=history_latents,
                                future_latents=future_latents,
                                epoch=self.epoch + 1,
                                save_dir=str(viz_dir),
                                trajectories=trajectories,
                                trajectories_base=trajectories_base,
                                trajectory_targets=trajectory_targets,
                            )
                            log.info(f"Saved latent visualizations to {viz_dir}")
                            log.info(f"  Avg t-SNE Distance: {viz_results['avg_tsne_distance']:.2f}")
                            
                            # Log metrics to wandb
                            wandb_metrics = {
                                "latent/avg_tsne_distance": viz_results['avg_tsne_distance'],
                            }
                            if 'flow_end_to_target_dist' in viz_results:
                                wandb_metrics["latent/flow_end_to_target_dist"] = viz_results['flow_end_to_target_dist']
                            if wandb_run is not None:
                                wandb_run.log(wandb_metrics, step=self.global_step)
                    except Exception as e:
                        log.warning(f"Failed to generate latent visualization: {e}")

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.train_config.training_params.sample_every) == 0:
                    if train_sampling_batch is None:
                        log.warning("Skip training sample visualization: no training batch available in this epoch.")
                    else:
                        with torch.no_grad():
                            # sample trajectory from training set, and evaluate difference
                            batch = train_sampling_batch
                            obs_dict = batch["obs"]
                            gt_action = batch["action"]

                            result = policy.predict_action(obs_dict)
                            pred_action = result["action_pred"]
                            
                            # Handle shape mismatch (e.g., VITA action-to-action flow outputs 8 frames from horizon=16)
                            pred_len = pred_action.shape[1]
                            gt_len = gt_action.shape[1]
                            if pred_len != gt_len:
                                # For action-to-action flow: pred is future actions starting from n_obs_steps-1
                                # Slice gt_action to match: take the corresponding future portion
                                n_obs_steps = gt_len - pred_len + 1  # Infer n_obs_steps from shape difference
                                start_idx = n_obs_steps - 1
                                gt_action = gt_action[:, start_idx:start_idx + pred_len, :]
                            
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            step_log["train_action_mse_error"] = mse.item()
                            del batch
                            del obs_dict
                            del gt_action
                            del result
                            del pred_action
                            del mse

                # checkpoint
                if (
                    (self.epoch + 1) % cfg.train_config.training_params.checkpoint_every
                ) == 0 or self.epoch + 1 >= cfg.train_config.training_params.num_epochs:
                    # checkpointing
                    save_name = pathlib.Path(self.cfg.dataset_config.zarr_path).stem
                    self.save_checkpoint(
                        cfg.checkpoint.save_root_dir
                        + f"/checkpoints/{self.epoch + 1}.ckpt"
                    , use_thread=False)  # TODO

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                json_logger.log(step_log)
                if wandb_run is not None:
                    wandb_run.log(step_log, step=self.global_step)
                self.global_step += 1
                self.epoch += 1

    def evaluate(self, ckpt_path=None):
        args = self.eval_args

        def _hb(msg: str):
            """Force-flushed heartbeat to locate hangs in long IsaacSim eval runs."""
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[EVAL-HB {ts}] {msg}", flush=True)
            log.info(f"[EVAL-HB] {msg}")

        # Enable timeout-guarded close to avoid IsaacSim shutdown hang
        import os
        os.environ["METASIM_FORCE_EXIT_ON_CLOSE"] = "1"
        os.environ.setdefault("METASIM_CLOSE_TIMEOUT_SEC", "8")
        _hb(
            f"evaluate() start | task={args.task} sim={args.sim} num_envs={args.num_envs} "
            f"gpu_id={args.gpu_id} render={getattr(args, 'render_mode', 'raytracing')}"
        )

        num_envs: int = args.num_envs
        log.info(f"Using GPU device: {args.gpu_id}")
        task_cls = get_task_class(args.task)

        # Observation/render configuration
        cfg_obs_type = getattr(self.cfg.eval_config.policy_runner.obs, "obs_type", None)
        shape_meta_obs = getattr(getattr(self.cfg, "shape_meta", None), "obs", None)
        # Keep camera enabled whenever the policy expects RGB observations.
        use_camera = bool(shape_meta_obs is not None and "head_cam" in shape_meta_obs)
        if cfg_obs_type == "joint_pos" and use_camera:
            log.info(
                "[EvalStage] obs_type=joint_pos but policy expects head_cam; keeping camera rendering enabled."
            )
        elif not use_camera:
            log.info(
                "[EvalStage] Policy does not use head_cam; camera rendering disabled for eval."
            )

        # Camera configuration
        if args.task in {"stack_cube", "pick_cube", "pick_butter"}:
            dp_camera = True
        else:
            dp_camera = args.task != "close_box"

        is_libero_dataset = "libero_90" in args.task

        if is_libero_dataset:
            dp_pos = (2.0, 0.0, 2)
        elif dp_camera:
            dp_pos = (1.0, 0.0, 0.75)
        else:
            dp_pos = (1.5, 0.0, 1.5)

        camera = None
        if use_camera:
            camera = PinholeCameraCfg(
                name="camera0",
                data_types=["rgb"],
                width=448,
                height=448,
                pos=dp_pos,
                look_at=(0.0, 0.0, 0.0),
            )

        # Lighting setup
        render_mode = getattr(args, 'render_mode', 'raytracing')
        if render_mode == "pathtracing":
            ceiling_main = 18000.0
            ceiling_corners = 8000.0
        else:
            ceiling_main = 12000.0
            ceiling_corners = 5000.0

        from metasim.scenario.lights import DiskLightCfg, SphereLightCfg
        lights = []
        if use_camera:
            lights = [
                DiskLightCfg(
                    name="ceiling_main",
                    intensity=ceiling_main,
                    color=(1.0, 1.0, 1.0),
                    radius=1.2,
                    pos=(0.0, 0.0, 2.8),
                    rot=(0.7071, 0.0, 0.0, 0.7071),
                ),
                SphereLightCfg(
                    name="ceiling_ne",
                    intensity=ceiling_corners,
                    color=(1.0, 1.0, 1.0),
                    radius=0.6,
                    pos=(1.0, 1.0, 2.5),
                ),
                SphereLightCfg(
                    name="ceiling_nw",
                    intensity=ceiling_corners,
                    color=(1.0, 1.0, 1.0),
                    radius=0.6,
                    pos=(-1.0, 1.0, 2.5),
                ),
                SphereLightCfg(
                    name="ceiling_sw",
                    intensity=ceiling_corners,
                    color=(1.0, 1.0, 1.0),
                    radius=0.6,
                    pos=(-1.0, -1.0, 2.5),
                ),
                SphereLightCfg(
                    name="ceiling_se",
                    intensity=ceiling_corners,
                    color=(1.0, 1.0, 1.0),
                    radius=0.6,
                    pos=(1.0, -1.0, 2.5),
                ),
            ]

        scenario = task_cls.scenario.update(
            robots=[args.robot],
            simulator=args.sim,
            num_envs=args.num_envs,
            headless=args.headless,
            render=RenderCfg(mode=render_mode),
            lights=lights,
            cameras=[camera] if use_camera else []
        )
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu_id}")
            torch.cuda.set_device(args.gpu_id)
        else:
            device = torch.device("cpu")
        _hb(f"torch device resolved: {device}")
        tic = time.time()
        log.info("[EvalStage] Creating task env...")
        _hb("about to create task env (task_cls(scenario, device))")
        env = task_cls(scenario, device=device)
        log.info("[EvalStage] Task env created.")
        _hb("task env created")
        robot = get_robot(args.robot)

        # Domain Randomization configuration
        dr_level = getattr(args, 'level', 0)
        dr_scene_mode = getattr(args, 'scene_mode', 0)
        dr_seed = getattr(args, 'randomization_seed', None)
        sim_name = getattr(args, "sim", "")

        if dr_level > 0 and sim_name != "isaacsim":
            # IsaacSim-independent fallback for environments like MuJoCo.
            randomization_manager = MujocoStateRandomizationManager(
                DRConfig(
                    level=dr_level,
                    scene_mode=dr_scene_mode,
                    randomization_seed=dr_seed,
                )
            )
            log.warning(
                f"Using MuJoCo fallback DR (state-pose perturbation only) for sim='{sim_name}', "
                f"level={dr_level}, scene_mode={dr_scene_mode}."
            )
        elif not RANDOMIZATION_AVAILABLE:
            if dr_level > 0:
                raise RuntimeError(
                    "Domain randomization requested but randomization components are unavailable. "
                    "Please install required dependencies or set dr_level_eval=0."
                )
            randomization_manager = None
        else:
            from dataclasses import dataclass as dc

            @dc
            class SimpleRenderCfg:
                mode: str = render_mode

            try:
                log.info("[EvalStage] Initializing DomainRandomizationManager...")
                _hb("about to initialize DomainRandomizationManager")
                randomization_manager = DomainRandomizationManager(
                    config=DRConfig(
                        level=dr_level,
                        scene_mode=dr_scene_mode,
                        randomization_seed=dr_seed,
                    ),
                    scenario=scenario,
                    handler=env.handler,
                    init_states=None,
                    render_cfg=SimpleRenderCfg(mode=render_mode)
                )
                log.info("[EvalStage] DomainRandomizationManager initialized.")
                _hb("DomainRandomizationManager initialized")
            except Exception as e:
                if dr_level > 0:
                    raise RuntimeError(
                        f"Domain randomization init failed with requested level={dr_level} "
                        f"(sim='{sim_name}'): {type(e).__name__}: {e}"
                    ) from e
                log.warning(
                    f"Domain randomization init failed ({type(e).__name__}: {e}). "
                    "Continuing with DR disabled because requested level=0."
                )
                randomization_manager = None
            if dr_level > 0:
                log.info(f"Domain Randomization enabled: level={dr_level}, scene_mode={dr_scene_mode}, seed={dr_seed}")
            else:
                log.info("Domain Randomization disabled (level=0)")

        toc = time.time()
        log.trace(f"Time to launch: {toc - tic:.2f}s")

        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint = self.get_checkpoint_path()
        checkpoint = ckpt_path if ckpt_path is not None else checkpoint
        if checkpoint is None:
            raise ValueError(
                "No checkpoint found, please provide a valid checkpoint path."
            )
        args.checkpoint_path = pathlib.Path(checkpoint)
        ckpt_name = args.checkpoint_path.name + "_" + time_str
        eval_tag = f"dr{dr_level}_scene{dr_scene_mode}"
        policy_cfg = getattr(self.cfg, "policy_config", None)
        flow_start_mode = str(getattr(policy_cfg, "flow_start_mode", "history"))
        flow_start_scale = float(getattr(policy_cfg, "flow_start_scale", 1.0))
        if flow_start_mode != "history":
            eval_tag += f"_start{flow_start_mode}"
            if flow_start_mode == "gaussian":
                flow_start_noise_std = float(getattr(policy_cfg, "flow_start_noise_std", 1.0))
                eval_tag += f"_std{flow_start_noise_std:.2f}"
        elif abs(flow_start_scale - 1.0) > 1e-8:
            eval_tag += f"_startscale{flow_start_scale:.2f}"
        if getattr(args, "mid_obj_perturb_enable", False):
            mx = float(getattr(args, "mid_obj_perturb_x", 0.10))
            my = float(getattr(args, "mid_obj_perturb_y", 0.0))
            mz = float(getattr(args, "mid_obj_perturb_z", 0.0))
            ms = int(getattr(args, "mid_obj_perturb_step", 40))
            names = str(getattr(args, "mid_obj_perturb_obj_names", "") or "")
            name_tag = ""
            if names:
                safe_names = names.replace(",", "-").replace("/", "_").replace(" ", "")
                name_tag = f"_{safe_names}"
            eval_tag += f"_midobj{name_tag}_t{ms}_dx{mx:.2f}_dy{my:.2f}_dz{mz:.2f}"
        if getattr(args, "probe_enable", False):
            pt = str(getattr(args, "probe_transform", "noise"))
            pm = str(getattr(args, "probe_mode", "continuous"))
            ps = int(getattr(args, "probe_step", 30))
            sd = getattr(args, "probe_seed", None)
            seed_tag = f"_s{sd}" if sd is not None else ""
            if pt == "noise":
                sigma = float(getattr(args, "probe_sigma", 0.0))
                eval_tag += f"_probe_{pt}_sig{sigma:.3f}_{pm}_t{ps}{seed_tag}"
            else:
                eval_tag += f"_probe_{pt}_{pm}_t{ps}{seed_tag}"
        if getattr(args, "pos_offset_enable", False):
            xm = float(getattr(args, "pos_offset_x_min", -0.05))
            xp = float(getattr(args, "pos_offset_x_max", 0.05))
            ym = float(getattr(args, "pos_offset_y_min", -0.05))
            yp = float(getattr(args, "pos_offset_y_max", 0.05))
            zm = float(getattr(args, "pos_offset_z_min", 0.0))
            zp = float(getattr(args, "pos_offset_z_max", 0.0))
            per_demo = "pd" if getattr(args, "pos_offset_per_demo", True) else "gl"
            sd = getattr(args, "pos_offset_seed", None)
            seed_tag = f"_s{sd}" if sd is not None else ""
            eval_tag += f"_poff_x[{xm:.2f},{xp:.2f}]_y[{ym:.2f},{yp:.2f}]_z[{zm:.2f},{zp:.2f}]_{per_demo}{seed_tag}"
        if getattr(args, "robot_init_noise_enable", False):
            rs = float(getattr(args, "robot_init_noise_sigma", 0.03))
            rpd = "pd" if getattr(args, "robot_init_noise_per_demo", True) else "gl"
            sd = getattr(args, "robot_init_noise_seed", None)
            seed_tag = f"_s{sd}" if sd is not None else ""
            eval_tag += f"_rinit_s{rs:.3f}_{rpd}{seed_tag}"
        ckpt_name = f"{args.task}/{self.policy_name}/{args.robot}/{eval_tag}/{ckpt_name}"

        if getattr(args, "probe_enable", False) or getattr(args, "action_shock_enable", False):
            from roboverse_learn.il.runners.noise_probe_eval_runner import NoiseProbeEvalRunner
            eval_runner_cls = NoiseProbeEvalRunner
        else:
            from roboverse_learn.il.runners.default_eval_runner import DefaultEvalRunner
            eval_runner_cls = DefaultEvalRunner

        policyRunner = eval_runner_cls(
            self,
            scenario=scenario,
            num_envs=num_envs,
            checkpoint_path=args.checkpoint_path,
            device=f"cuda:{args.gpu_id}",
            task_name=args.task,
            subset=args.subset,
            probe_enable=getattr(args, "probe_enable", False),
            probe_sigma=getattr(args, "probe_sigma", 0.0),
            probe_transform=getattr(args, "probe_transform", "noise"),
            probe_mode=getattr(args, "probe_mode", "continuous"),
            probe_step=getattr(args, "probe_step", 30),
            probe_seed=getattr(args, "probe_seed", None),
            action_shock_enable=getattr(args, "action_shock_enable", False),
            action_shock_sigma=getattr(args, "action_shock_sigma", 0.0),
            action_shock_mode=getattr(args, "action_shock_mode", "one_shot"),
            action_shock_step=getattr(args, "action_shock_step", 30),
            action_shock_seed=getattr(args, "action_shock_seed", None),
            action_shock_clip=getattr(args, "action_shock_clip", True),
        )

        action_set_steps = (
            2 if policyRunner.policy_cfg.action_config.action_type == "ee" else 1
        )
        # Data
        tic = time.time()
        assert os.path.exists(env.traj_filepath), (
            f"Trajectory file: {env.traj_filepath} does not exist."
        )
        _hb(f"loading trajectory file: {env.traj_filepath}")
        init_states, all_actions, all_states = get_traj(env.traj_filepath, robot, env.handler)
        num_demos = len(init_states)
        toc = time.time()
        log.trace(f"Time to load data: {toc - tic:.2f}s")
        _hb(f"trajectory loaded | demos={num_demos}")

        # Object position randomization for spatial generalization testing.
        # Independent from visual DR level system. Disabled by default.
        pos_randomizer = None
        if getattr(args, "pos_offset_enable", False):
            from roboverse_learn.il.utils.position_randomizer import PositionRandomizer

            raw_names = str(getattr(args, "pos_offset_obj_names", "") or "")
            obj_names = [n.strip() for n in raw_names.split(",") if n.strip()] if raw_names else None

            pos_randomizer = PositionRandomizer(
                x_range=(
                    float(getattr(args, "pos_offset_x_min", -0.05)),
                    float(getattr(args, "pos_offset_x_max", 0.05)),
                ),
                y_range=(
                    float(getattr(args, "pos_offset_y_min", -0.05)),
                    float(getattr(args, "pos_offset_y_max", 0.05)),
                ),
                z_range=(
                    float(getattr(args, "pos_offset_z_min", 0.0)),
                    float(getattr(args, "pos_offset_z_max", 0.0)),
                ),
                obj_names=obj_names,
                per_demo=bool(getattr(args, "pos_offset_per_demo", True)),
                seed=getattr(args, "pos_offset_seed", None),
            )
            log.info(
                f"Position randomization enabled: x={pos_randomizer.x_range}, "
                f"y={pos_randomizer.y_range}, z={pos_randomizer.z_range}, "
                f"per_demo={pos_randomizer.per_demo}"
            )

        # Robot initial state perturbation (joint position noise).
        # Independent from visual DR and object position offset.
        robot_init_randomizer = None
        if getattr(args, "robot_init_noise_enable", False):
            from roboverse_learn.il.utils.robot_init_randomizer import RobotInitRandomizer
            robot_init_randomizer = RobotInitRandomizer(
                sigma=float(getattr(args, "robot_init_noise_sigma", 0.03)),
                per_demo=bool(getattr(args, "robot_init_noise_per_demo", True)),
                seed=getattr(args, "robot_init_noise_seed", None),
                clip=getattr(args, "robot_init_noise_clip", None),
            )
            log.info(f"Robot init perturbation enabled: sigma={robot_init_randomizer.sigma}")

        # Update DR manager with init_states
        if randomization_manager is not None:
            randomization_manager.init_states = init_states
            randomization_manager.original_positions = {}
            for demo_idx, init_state in enumerate(init_states):
                demo_key = f"demo_{demo_idx}"
                randomization_manager.original_positions[demo_key] = {}

                if "objects" in init_state:
                    for obj_name, obj_state in init_state["objects"].items():
                        randomization_manager.original_positions[demo_key][f"obj_{obj_name}"] = {
                            "x": float(obj_state["pos"][0]),
                            "y": float(obj_state["pos"][1]),
                            "z": float(obj_state["pos"][2]),
                        }

                if "robots" in init_state:
                    for robot_name, robot_state in init_state["robots"].items():
                        randomization_manager.original_positions[demo_key][f"robot_{robot_name}"] = {
                            "x": float(robot_state["pos"][0]),
                            "y": float(robot_state["pos"][1]),
                            "z": float(robot_state["pos"][2]),
                        }

        total_success = 0
        total_completed = 0
        all_inference_times = []  # Collect inference times from all steps
        demo_avg_inference_times = []  # Collect average inference time for each demo
        eval_demo_records = []
        
        if args.max_demo is None:
            max_demos = args.task_id_range_high - args.task_id_range_low
        else:
            max_demos = args.max_demo
        max_demos = min(max_demos, num_demos)
        _hb(
            f"eval loop start | demo_range=[{args.task_id_range_low}, {args.task_id_range_low + max_demos}) "
            f"step_limit={args.max_step}"
        )

        for demo_start_idx in range(
            args.task_id_range_low, args.task_id_range_low + max_demos, num_envs
        ):
            demo_end_idx = min(demo_start_idx + num_envs, num_demos)
            current_demo_idxs = list(range(demo_start_idx, demo_end_idx))
            _hb(f"demo batch start | demos={demo_start_idx}-{demo_end_idx - 1}")

            # Apply domain randomization before reset
            if randomization_manager is not None and dr_level > 0:
                for env_id, demo_idx in enumerate(current_demo_idxs):
                    log.info(f"[DP Eval] Episode {demo_idx}: Applying DR")
                    randomization_manager.apply_randomization(
                        demo_idx=demo_idx, is_initial=(demo_start_idx == args.task_id_range_low))
                    randomization_manager.update_positions_to_table(demo_idx=demo_idx, env_id=env_id)
                    randomization_manager.update_camera_look_at(env_id=env_id)
                    randomization_manager.apply_camera_randomization()

            tic = time.time()
            log.info(f"[EvalStage] Resetting env for demos {demo_start_idx}-{demo_end_idx}...")

            # Apply robot init perturbation (joint noise) then object position offset.
            # Both are opt-in and orthogonal. Robot init goes first so pos offset
            # operates on the perturbed-robot states.
            current_init_states = init_states
            if robot_init_randomizer is not None:
                current_init_states = robot_init_randomizer.apply(
                    current_init_states, demo_indices=current_demo_idxs
                )
            if pos_randomizer is not None:
                current_init_states = pos_randomizer.apply(
                    current_init_states, demo_indices=current_demo_idxs
                )

            _hb(f"about to env.reset(states=current_init_states[{demo_start_idx}:{demo_end_idx}])")
            obs, extras = env.reset(states=current_init_states[demo_start_idx:demo_end_idx])
            log.info(f"[EvalStage] Reset done for demos {demo_start_idx}-{demo_end_idx}.")
            _hb(f"env.reset done | cost={time.time() - tic:.2f}s")
            toc = time.time()
            log.trace(f"Time to reset: {toc - tic:.2f}s")

            # Ensure environment stabilizes after reset
            if randomization_manager is not None and dr_level > 0:
                ensure_clean_state(env.handler)

                if hasattr(env, "_episode_steps"):
                    for env_id in range(num_envs):
                        env._episode_steps[env_id] = 0

            policyRunner.reset()

            step = 0
            MaxStep = args.max_step
            SuccessOnce = [False] * num_envs
            TimeOut = [False] * num_envs
            images_list = []
            inference_times = []  # Record inference time for each step
            mid_obj_perturbed = False
            print(policyRunner.policy_cfg)

            while step < MaxStep:
                if step == 0:
                    log.info(f"[EvalStage] Entering step loop for demos {demo_start_idx}-{demo_end_idx}.")
                    _hb(f"entered step loop | max_step={MaxStep}")
                elif step % 25 == 0:
                    _hb(f"step heartbeat | step={step}/{MaxStep}")
                if (
                    getattr(args, "mid_obj_perturb_enable", False)
                    and (not mid_obj_perturbed)
                    and step >= int(getattr(args, "mid_obj_perturb_step", 40))
                ):
                    obs = apply_mid_rollout_object_perturbation(env, args, num_envs)
                    if hasattr(policyRunner, "action_cache"):
                        policyRunner.action_cache.clear()
                    mid_obj_perturbed = True
                    _hb(f"mid-rollout object perturbation applied at step={step}")
                if use_camera:
                    obs_rgb = obs.cameras["camera0"].rgb
                else:
                    obs_rgb = torch.zeros(
                        (num_envs, 256, 256, 3),
                        dtype=torch.uint8,
                        device=obs.robots[args.robot].joint_pos.device,
                    )
                new_obs = {
                    "rgb": obs_rgb,
                    "joint_qpos": obs.robots[args.robot].joint_pos,
                }

                images_list.append(np.array(new_obs["rgb"].cpu()))
                
                # Measure inference time
                inference_start = time.time()
                action = policyRunner.get_action(new_obs)
                inference_end = time.time()
                inference_time_ms = (inference_end - inference_start) * 1000
                inference_times.append(inference_time_ms)
                
                log.debug(f"Step {step} | Inference time: {inference_time_ms:.2f}ms")

                for round_i in range(action_set_steps):
                    obs, reward, success, time_out, extras = env.step(action)

                # eval: convert to python bools to avoid tensor(bool) bookkeeping issues.
                success_flags = (
                    success.detach().cpu().tolist()
                    if isinstance(success, torch.Tensor)
                    else list(success)
                )
                timeout_flags = (
                    time_out.detach().cpu().tolist()
                    if isinstance(time_out, torch.Tensor)
                    else list(time_out)
                )
                SuccessOnce = [SuccessOnce[i] or bool(success_flags[i]) for i in range(num_envs)]
                TimeOut = [TimeOut[i] or bool(timeout_flags[i]) for i in range(num_envs)]
                step += 1
                if all(SuccessOnce):
                    _hb(f"all envs succeeded early at step={step}")
                    break

            # Calculate inference time statistics
            total_steps = len(inference_times)
            avg_inference_time = sum(inference_times) / total_steps if total_steps > 0 else 0
            min_inference_time = min(inference_times) if inference_times else 0
            max_inference_time = max(inference_times) if inference_times else 0
            
            log.info(f"Demo {demo_start_idx}-{demo_end_idx}: Avg inference time: {avg_inference_time:.2f}ms, "
                     f"Min: {min_inference_time:.2f}ms, Max: {max_inference_time:.2f}ms, Total steps: {total_steps}")
            
            # Collect inference times for overall statistics
            all_inference_times.extend(inference_times)
            demo_avg_inference_times.append(avg_inference_time)  # Store demo-level average

            SuccessEnd = [bool(x) for x in success_flags]
            total_success += SuccessOnce.count(True)
            total_completed += len(SuccessOnce)
            base_eval_dir = pathlib.Path(self.output_dir).joinpath("eval", ckpt_name)
            try:
                base_eval_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                _hb(f"FAILED creating eval dir: {base_eval_dir} | err={type(e).__name__}: {e}")
                raise
            _hb(f"writing demo outputs to: {base_eval_dir}")
            for i, demo_idx in enumerate(range(demo_start_idx, demo_end_idx)):
                demo_idx_str = str(demo_idx).zfill(4)
                if i % args.save_video_freq == 0:
                    iio.mimwrite(
                        str(base_eval_dir.joinpath(f"{demo_idx}.mp4")),
                        [images[i] for images in images_list],
                    )
                with open(base_eval_dir.joinpath(f"{demo_idx_str}.txt"), "w") as f:
                    f.write(f"Demo Index: {demo_idx}\n")
                    f.write(f"Num Envs: {num_envs}\n")
                    f.write(f"SuccessOnce: {SuccessOnce[i]}\n")
                    f.write(f"SuccessEnd: {SuccessEnd[i]}\n")
                    f.write(f"TimeOut: {TimeOut[i]}\n")
                    f.write(f"Domain Randomization Level: {dr_level}\n")
                    f.write(f"Domain Randomization Scene Mode: {dr_scene_mode}\n")
                    f.write(f"Domain Randomization Seed: {dr_seed}\n")
                    if pos_randomizer is not None:
                        dx, dy, dz = pos_randomizer._get_offset(demo_idx)
                        f.write(f"Position Offset (x, y, z): ({dx:.4f}, {dy:.4f}, {dz:.4f})\n")
                    if getattr(args, "mid_obj_perturb_enable", False):
                        f.write(f"Mid Object Perturb Enabled: True\n")
                        f.write(f"Mid Object Perturb Applied: {mid_obj_perturbed}\n")
                        f.write(f"Mid Object Perturb Step: {int(getattr(args, 'mid_obj_perturb_step', 40))}\n")
                        f.write(
                            "Mid Object Perturb Offset (x, y, z): "
                            f"({float(getattr(args, 'mid_obj_perturb_x', 0.10)):.4f}, "
                            f"{float(getattr(args, 'mid_obj_perturb_y', 0.0)):.4f}, "
                            f"{float(getattr(args, 'mid_obj_perturb_z', 0.0)):.4f})\n"
                        )
                        f.write(
                            f"Mid Object Perturb Obj Names: {str(getattr(args, 'mid_obj_perturb_obj_names', '') or 'ALL')}\n"
                        )
                    if robot_init_randomizer is not None:
                        f.write(f"Robot Init Noise Enabled: True\n")
                        f.write(f"Robot Init Noise Sigma: {robot_init_randomizer.sigma:.4f}\n")
                    f.write(
                        f"Cumulative Average Success Rate: {total_success / total_completed:.4f}\n"
                    )
                    # Add inference time statistics
                    f.write(f"\n--- Inference Time Statistics ---\n")
                    f.write(f"Total Steps: {total_steps}\n")
                    f.write(f"Average Inference Time: {avg_inference_time:.2f}ms\n")
                    f.write(f"Min Inference Time: {min_inference_time:.2f}ms\n")
                    f.write(f"Max Inference Time: {max_inference_time:.2f}ms\n")
                eval_demo_records.append(
                    {
                        "demo_index": int(demo_idx),
                        "success_once": bool(SuccessOnce[i]),
                        "success_end": bool(SuccessEnd[i]),
                        "timeout": bool(TimeOut[i]),
                        "total_steps": int(total_steps),
                        "avg_infer_ms": float(avg_inference_time),
                        "min_infer_ms": float(min_inference_time),
                        "max_infer_ms": float(max_inference_time),
                        "mid_obj_perturb_applied": bool(mid_obj_perturbed),
                    }
                )
            _hb(f"demo batch finished | demos={demo_start_idx}-{demo_end_idx - 1}")
            log.info("Demo Indices: ", range(demo_start_idx, demo_end_idx))
            log.info("Num Envs: ", num_envs)
            log.info(f"SuccessOnce: {SuccessOnce}")
            log.info(f"SuccessEnd: {SuccessEnd}")
            log.info(f"TimeOut: {TimeOut}")
        # Calculate overall inference time statistics
        overall_total_steps = len(all_inference_times)
        overall_avg_inference_time = sum(all_inference_times) / overall_total_steps if overall_total_steps > 0 else 0
        overall_min_inference_time = min(all_inference_times) if all_inference_times else 0
        overall_max_inference_time = max(all_inference_times) if all_inference_times else 0
        
        # Calculate STD of demo-level average inference times
        num_demos_evaluated = len(demo_avg_inference_times)
        if num_demos_evaluated > 1:
            demo_avg_mean = sum(demo_avg_inference_times) / num_demos_evaluated
            demo_avg_variance = sum((x - demo_avg_mean) ** 2 for x in demo_avg_inference_times) / (num_demos_evaluated - 1)
            demo_avg_std = demo_avg_variance ** 0.5
        else:
            demo_avg_std = 0.0
        
        success_rate = total_success / total_completed
        log.info(f"FINAL RESULTS: Average Success Rate = {success_rate:.4f}")
        log.info(f"FINAL RESULTS: Overall Avg Inference Time = {overall_avg_inference_time:.2f}ms (STD across demos: {demo_avg_std:.2f}ms), "
                 f"Min: {overall_min_inference_time:.2f}ms, Max: {overall_max_inference_time:.2f}ms, "
                 f"Total Steps: {overall_total_steps}")
        
        with open(base_eval_dir.joinpath("final_stats.txt"), "w") as f:
            f.write(f"=== Success Statistics ===\n")
            f.write(f"Total Success: {total_success}\n")
            f.write(f"Total Completed: {total_completed}\n")
            f.write(f"Average Success Rate: {success_rate:.4f}\n")
            f.write(f"\n=== Domain Randomization ===\n")
            f.write(f"Domain Randomization Level: {dr_level}\n")
            f.write(f"Domain Randomization Scene Mode: {dr_scene_mode}\n")
            f.write(f"Domain Randomization Seed: {dr_seed}\n")
            if getattr(args, "mid_obj_perturb_enable", False):
                f.write(f"\n=== Mid-Rollout Object Perturbation ===\n")
                f.write(f"Enabled: True\n")
                f.write(f"Step: {int(getattr(args, 'mid_obj_perturb_step', 40))}\n")
                f.write(
                    "Offset (x, y, z): "
                    f"({float(getattr(args, 'mid_obj_perturb_x', 0.10)):.4f}, "
                    f"{float(getattr(args, 'mid_obj_perturb_y', 0.0)):.4f}, "
                    f"{float(getattr(args, 'mid_obj_perturb_z', 0.0)):.4f})\n"
                )
                f.write(f"Object Names: {str(getattr(args, 'mid_obj_perturb_obj_names', '') or 'ALL')}\n")
            f.write(f"\n=== Overall Inference Time Statistics ===\n")
            f.write(f"Total Inference Steps: {overall_total_steps}\n")
            f.write(f"Number of Demos Evaluated: {num_demos_evaluated}\n")
            f.write(f"Average Inference Time: {overall_avg_inference_time:.2f}ms\n")
            f.write(f"STD of Demo Avg Inference Time: {demo_avg_std:.2f}ms\n")
            f.write(f"Min Inference Time: {overall_min_inference_time:.2f}ms\n")
            f.write(f"Max Inference Time: {overall_max_inference_time:.2f}ms\n")

        eval_args_dict = {k: _json_safe(v) for k, v in vars(args).items()}
        summary_payload = {
            "schema_version": 1,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(base_eval_dir),
            "output_dir": str(self.output_dir),
            "task": str(args.task),
            "policy": str(self.policy_name),
            "robot": str(args.robot),
            "checkpoint_path": str(args.checkpoint_path),
            "eval_tag": eval_tag,
            "eval_args": eval_args_dict,
            "perturbations": {
                "domain_randomization": {
                    "level": int(dr_level),
                    "scene_mode": int(dr_scene_mode),
                    "seed": _json_safe(dr_seed),
                },
                "position_offset": {
                    "enabled": bool(getattr(args, "pos_offset_enable", False)),
                    "x_min": float(getattr(args, "pos_offset_x_min", -0.05)),
                    "x_max": float(getattr(args, "pos_offset_x_max", 0.05)),
                    "y_min": float(getattr(args, "pos_offset_y_min", -0.05)),
                    "y_max": float(getattr(args, "pos_offset_y_max", 0.05)),
                    "z_min": float(getattr(args, "pos_offset_z_min", 0.0)),
                    "z_max": float(getattr(args, "pos_offset_z_max", 0.0)),
                    "obj_names": str(getattr(args, "pos_offset_obj_names", "") or "ALL"),
                    "per_demo": bool(getattr(args, "pos_offset_per_demo", True)),
                    "seed": _json_safe(getattr(args, "pos_offset_seed", None)),
                },
                "mid_rollout_object": {
                    "enabled": bool(getattr(args, "mid_obj_perturb_enable", False)),
                    "step": int(getattr(args, "mid_obj_perturb_step", 40)),
                    "x": float(getattr(args, "mid_obj_perturb_x", 0.10)),
                    "y": float(getattr(args, "mid_obj_perturb_y", 0.0)),
                    "z": float(getattr(args, "mid_obj_perturb_z", 0.0)),
                    "obj_names": str(getattr(args, "mid_obj_perturb_obj_names", "") or "ALL"),
                    "zero_vel": bool(getattr(args, "mid_obj_perturb_zero_vel", True)),
                },
                "history_probe": {
                    "enabled": bool(getattr(args, "probe_enable", False)),
                    "transform": str(getattr(args, "probe_transform", "noise")),
                    "mode": str(getattr(args, "probe_mode", "continuous")),
                    "sigma": float(getattr(args, "probe_sigma", 0.0)),
                    "step": int(getattr(args, "probe_step", 30)),
                    "seed": _json_safe(getattr(args, "probe_seed", None)),
                },
            },
            "results": {
                "total_success": int(total_success),
                "total_completed": int(total_completed),
                "success_rate": float(success_rate),
                "success_once": int(total_success),
                "num_demos_evaluated": int(num_demos_evaluated),
                "overall_total_steps": int(overall_total_steps),
                "overall_avg_infer_ms": float(overall_avg_inference_time),
                "demo_avg_infer_std_ms": float(demo_avg_std),
                "overall_min_infer_ms": float(overall_min_inference_time),
                "overall_max_infer_ms": float(overall_max_inference_time),
            },
            "demos": eval_demo_records,
        }
        _write_json(base_eval_dir.joinpath("eval_summary.json"), summary_payload)
        summary_jsonl = pathlib.Path(self.output_dir).joinpath("eval_results.jsonl")
        with open(summary_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(summary_payload), sort_keys=True) + "\n")
        _hb(f"wrote eval summary json: {base_eval_dir.joinpath('eval_summary.json')}")
        _hb(f"appended eval summary jsonl: {summary_jsonl}")
        _hb("about to env.close()")
        env.close()
        _hb("env.close() returned; evaluate() done")

    def run(
        self,
        train=None,
        eval=None,
        ckpt_path=None,
    ):
        train = self.cfg.train_enable
        eval = self.cfg.eval_enable
        # Always use eval_path if provided (respects num_epochs setting)
        ckpt_path = self.cfg.eval_path
        if train:
            self.train()
        if eval:
            self.evaluate(ckpt_path=ckpt_path)


class BatchSampler:
    def __init__(
        self,
        data_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[: -self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int = 0,
):
    # print("create_dataloader_batch_size", batch_size)
    batch_sampler = BatchSampler(
        len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True
    )

    def collate(x):
        assert len(x) == 1
        return x[0]

    dataloader = DataLoader(
        dataset,
        collate_fn=collate,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )
    return dataloader


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = DefaultRunner(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
