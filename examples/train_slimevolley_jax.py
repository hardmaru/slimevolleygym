"""Train shared-parameter IPPO on Slime Volleyball (JAX backend).

Demonstrates the dual-backend JAX-native step/reset pipeline with:
- jax.vmap for parallel environments
- jax.lax.scan for the training loop
- Auto-reset on episode completion

Adapted from the JaxMARL / CoGrid IPPO training script.

Usage:
    python examples/train_slimevolley_jax.py
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple
from flax.training.train_state import TrainState

from slime_volleyball.backend._dispatch import _reset_backend_for_testing
from slime_volleyball.backend import set_backend
from slime_volleyball.slimevolley_env import SlimeVolleyEnv


# ---- Action table (discrete -> [fwd, bwd, jump]) ----

ACTION_TABLE = jnp.array([
    [0, 0, 0],  # NOOP
    [1, 0, 0],  # LEFT (forward)
    [1, 0, 1],  # UPLEFT (forward jump)
    [0, 0, 1],  # UP (jump)
    [0, 1, 1],  # UPRIGHT (backward jump)
    [0, 1, 0],  # RIGHT (backward)
], dtype=jnp.float32)


# ---- Categorical distribution helpers ----


def categorical_sample(rng, logits):
    return jax.random.categorical(rng, logits)


def categorical_log_prob(logits, actions):
    log_probs = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    return jnp.take_along_axis(log_probs, actions[..., None].astype(jnp.int32), axis=-1).squeeze(-1)


def categorical_entropy(logits):
    log_probs = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)


# ---- Network ----


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        actor = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor = activation(actor)
        actor = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor)
        actor = activation(actor)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)

        critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return logits, jnp.squeeze(critic, axis=-1)


# ---- Transition storage ----


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


# ---- Training ----


def make_train(config, step_fn, reset_fn, n_agents, n_actions, obs_dim):
    """Build a fully JIT-compilable IPPO train function."""
    num_envs = config["NUM_ENVS"]
    num_actors = n_agents * num_envs
    num_steps = config["NUM_STEPS"]
    num_updates = int(config["TOTAL_TIMESTEPS"] // num_steps // num_envs)
    num_minibatches = config["NUM_MINIBATCHES"]

    network = ActorCritic(n_actions, activation=config["ACTIVATION"])

    def linear_schedule(count):
        frac = 1.0 - (count // (num_minibatches * config["UPDATE_EPOCHS"])) / num_updates
        return config["LR"] * frac

    def train(rng):
        # ---- Init network ----
        rng, init_rng = jax.random.split(rng)
        params = network.init(init_rng, jnp.zeros(obs_dim, dtype=jnp.float32))

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        # ---- Init envs (vmapped) ----
        rng, reset_rng = jax.random.split(rng)
        obs, env_state, _ = jax.vmap(reset_fn)(jax.random.split(reset_rng, num_envs))
        # obs: (NUM_ENVS, n_agents, obs_dim)

        ep_return = jnp.zeros(num_envs, dtype=jnp.float32)
        ep_length = jnp.zeros(num_envs, dtype=jnp.int32)

        # ---- Outer loop: one iteration = collect + update ----
        def _update_step(runner_state, unused):
            def _env_step(carry, unused):
                train_state, env_state, last_obs, ep_return, ep_length, rng = carry

                # Batchify: (NUM_ENVS, n_agents, obs_dim) -> (num_actors, obs_dim)
                obs_batch = last_obs.reshape(num_actors, -1).astype(jnp.float32)

                # Forward pass (shared params for all agents)
                rng, action_rng = jax.random.split(rng)
                logits, value = network.apply(train_state.params, obs_batch)
                action = categorical_sample(action_rng, logits)
                log_prob = categorical_log_prob(logits, action)

                # Unbatchify discrete actions -> continuous: (num_actors,) -> (NUM_ENVS, n_agents, 3)
                env_actions_discrete = action.reshape(num_envs, n_agents)
                env_actions = ACTION_TABLE[env_actions_discrete]  # (NUM_ENVS, n_agents, 3)

                # Step all envs in parallel
                new_obs, new_state, rewards, terms, truncs, _ = jax.vmap(step_fn)(
                    env_state, env_actions
                )

                # Add survival reward: each agent gets +0.01 per step
                rewards = rewards + config["SURVIVAL_REWARD"]

                done = terms | truncs  # (NUM_ENVS, n_agents)
                any_done = jnp.any(done, axis=-1)  # (NUM_ENVS,)

                # Track right agent's return (agent index 1)
                new_ep_return = ep_return + rewards[:, 1]
                # Also track episode length via step counter
                new_ep_length = ep_length + 1
                returned_ep_return = jnp.where(any_done, new_ep_return, 0.0)
                returned_ep_length = jnp.where(any_done, new_ep_length, jnp.int32(0))
                returned_episode = any_done.astype(jnp.float32)
                ep_return_next = jnp.where(any_done, 0.0, new_ep_return)
                ep_length_next = jnp.where(any_done, jnp.int32(0), new_ep_length)

                # Auto-reset done envs
                rng, reset_rng = jax.random.split(rng)
                reset_obs, reset_state, _ = jax.vmap(reset_fn)(
                    jax.random.split(reset_rng, num_envs)
                )

                def _select(reset_val, step_val):
                    shape = (num_envs,) + (1,) * (reset_val.ndim - 1)
                    return jnp.where(any_done.reshape(shape), reset_val, step_val)

                final_state = jax.tree.map(_select, reset_state, new_state)
                final_obs = _select(reset_obs, new_obs)

                transition = Transition(
                    done=done.reshape(num_actors).astype(jnp.float32),
                    action=action,
                    value=value,
                    reward=rewards.reshape(num_actors),
                    log_prob=log_prob,
                    obs=obs_batch,
                )
                carry = (
                    train_state,
                    final_state,
                    final_obs,
                    ep_return_next,
                    ep_length_next,
                    rng,
                )
                return carry, (transition, returned_ep_return, returned_ep_length, returned_episode)

            # Collect trajectories
            carry, (traj_batch, ep_returns, ep_lengths, ep_dones) = jax.lax.scan(
                _env_step, runner_state, None, num_steps
            )
            train_state, env_state, last_obs, ep_return, ep_length, rng = carry

            # ---- GAE ----
            last_obs_batch = last_obs.reshape(num_actors, -1).astype(jnp.float32)
            _, last_val = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # ---- PPO update epochs ----
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        logits, value = network.apply(params, traj_batch.obs)
                        log_prob = categorical_log_prob(logits, traj_batch.action)
                        entropy = categorical_entropy(logits)

                        # Value loss (clipped)
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(
                                jnp.square(value - targets),
                                jnp.square(value_pred_clipped - targets),
                            ).mean()
                        )

                        # Actor loss (clipped surrogate)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae,
                        ).mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy.mean()
                        )
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy.mean(),
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                batch_size = num_actors * num_steps
                permutation = jax.random.permutation(perm_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree.map(
                    lambda x: x.reshape((num_minibatches, -1) + x.shape[1:]),
                    shuffled,
                )
                train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)
                return (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ), total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            metrics = {
                "returned_episode_returns": ep_returns,
                "returned_episode_lengths": ep_lengths,
                "returned_episode": ep_dones,
            }
            runner_state = (train_state, env_state, last_obs, ep_return, ep_length, rng)
            return runner_state, metrics

        rng, train_rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obs, ep_return, ep_length, train_rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, num_updates)
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def visualize_policy(
    network,
    params,
    max_steps=3000,
    gif_path="examples/episode.gif",
    fps=30,
    seed=0,
):
    """Roll out trained policy and save as GIF.

    Uses the numpy backend for rendering via the legacy game engine,
    while applying the trained JAX policy for action selection.
    """
    import imageio
    from slime_volleyball.backend._dispatch import _reset_backend_for_testing
    from slime_volleyball.core import constants
    from slime_volleyball.core.observations import compute_observations

    # We render using the legacy OOP game engine (numpy only).
    # Build a fresh game and step it manually, selecting actions with the
    # trained network (which lives in JAX arrays).
    _reset_backend_for_testing()
    constants.setPixelObsMode()

    from slime_volleyball.core.game import SlimeVolleyGame
    from slime_volleyball.core import utils

    rng = np.random.default_rng(seed)
    game = SlimeVolleyGame(rng)

    action_table_np = np.array([
        [0, 0, 0],  # NOOP
        [1, 0, 0],  # LEFT
        [1, 0, 1],  # UPLEFT
        [0, 0, 1],  # UP
        [0, 1, 1],  # UPRIGHT
        [0, 1, 0],  # RIGHT
    ], dtype=np.float32)

    frames = []
    jax_rng = jax.random.PRNGKey(seed)

    for step_i in range(max_steps):
        # Render current state
        canvas = game.display(None)
        canvas = utils.downsize_image(canvas)
        # Convert BGR (cv2) to RGB for GIF
        frame = canvas[:, :, ::-1].copy()
        frames.append(frame)

        # Get observations for both agents (perspective-normalized)
        obs_left = game.agent_left.get_observation()
        obs_right = game.agent_right.get_observation()
        obs_array = np.stack([obs_left, obs_right], axis=0)  # (2, 12)

        # Select actions with trained policy
        logits, _ = network.apply(params, obs_array)
        jax_rng, act_rng = jax.random.split(jax_rng)
        actions = jax.random.categorical(act_rng, logits)  # (2,)

        left_action = action_table_np[int(actions[0])]
        right_action = action_table_np[int(actions[1])]

        game.agent_left.set_action(left_action)
        game.agent_right.set_action(right_action)

        result = game.step()

        if result != 0:
            # Goal scored — render the final frame
            canvas = game.display(None)
            canvas = utils.downsize_image(canvas)
            frames.append(canvas[:, :, ::-1].copy())
            break

    # Stamp a 2px progress bar so GIF encoder keeps every frame
    for i, frame in enumerate(frames):
        fill = max(1, int((i / max(len(frames), 1)) * frame.shape[1]))
        frame[-2:, :fill] = [60, 60, 60]

    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"Saved {len(frames)}-frame GIF to {gif_path}")


def export_to_onnx(params, obs_dim, n_actions, activation="tanh", onnx_path="examples/policy.onnx"):
    """Export the trained ActorCritic Flax model to ONNX format.

    Builds the ONNX graph directly from Flax parameters. Exports both the
    actor (logits) and critic (value) heads.

    Args:
        params: Flax parameter dict (the 'params' key from TrainState).
        obs_dim: Observation vector size.
        n_actions: Number of discrete actions (actor output size).
        activation: "tanh" or "relu".
        onnx_path: Output file path for the .onnx model.
    """
    import onnx
    from onnx import TensorProto, helper

    p = params["params"]

    # Build initializers (weights & biases) from Flax params
    initializers = []
    for name in ["Dense_0", "Dense_1", "Dense_2", "Dense_3", "Dense_4", "Dense_5"]:
        kernel = np.array(p[name]["kernel"])
        bias = np.array(p[name]["bias"])
        initializers.append(
            helper.make_tensor(
                f"{name}_weight", TensorProto.FLOAT, kernel.shape, kernel.flatten().tolist()
            )
        )
        initializers.append(
            helper.make_tensor(
                f"{name}_bias", TensorProto.FLOAT, bias.shape, bias.flatten().tolist()
            )
        )

    act_type = "Tanh" if activation == "tanh" else "Relu"

    # Actor head: Dense_0 -> act -> Dense_1 -> act -> Dense_2
    actor_nodes = [
        helper.make_node(
            "Gemm", ["input", "Dense_0_weight", "Dense_0_bias"], ["actor_h0"], transB=0
        ),
        helper.make_node(act_type, ["actor_h0"], ["actor_a0"]),
        helper.make_node(
            "Gemm", ["actor_a0", "Dense_1_weight", "Dense_1_bias"], ["actor_h1"], transB=0
        ),
        helper.make_node(act_type, ["actor_h1"], ["actor_a1"]),
        helper.make_node(
            "Gemm", ["actor_a1", "Dense_2_weight", "Dense_2_bias"], ["logits"], transB=0
        ),
    ]

    # Critic head: Dense_3 -> act -> Dense_4 -> act -> Dense_5 -> squeeze
    squeeze_axes = helper.make_tensor("squeeze_axes", TensorProto.INT64, [1], [1])
    initializers.append(squeeze_axes)

    critic_nodes = [
        helper.make_node(
            "Gemm", ["input", "Dense_3_weight", "Dense_3_bias"], ["critic_h0"], transB=0
        ),
        helper.make_node(act_type, ["critic_h0"], ["critic_a0"]),
        helper.make_node(
            "Gemm", ["critic_a0", "Dense_4_weight", "Dense_4_bias"], ["critic_h1"], transB=0
        ),
        helper.make_node(act_type, ["critic_h1"], ["critic_a1"]),
        helper.make_node(
            "Gemm", ["critic_a1", "Dense_5_weight", "Dense_5_bias"], ["value_2d"], transB=0
        ),
        helper.make_node("Squeeze", ["value_2d", "squeeze_axes"], ["value"]),
    ]

    graph = helper.make_graph(
        actor_nodes + critic_nodes,
        "ActorCritic",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, obs_dim])],
        outputs=[
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, n_actions]),
            helper.make_tensor_value_info("value", TensorProto.FLOAT, [None]),
        ],
        initializer=initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    print(f"Exported ONNX model to {onnx_path}")

    # Verify with onnxruntime
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    test_input = np.random.randn(1, obs_dim).astype(np.float32)
    ort_logits, ort_value = sess.run(None, {"input": test_input})
    print(f"  ONNX verification passed — logits: {ort_logits.shape}, value: {ort_value.shape}")


if __name__ == "__main__":
    _reset_backend_for_testing()
    set_backend("jax")

    config = {
        "LR": 3e-4,
        "NUM_ENVS": 64,
        "NUM_STEPS": 256,
        "TOTAL_TIMESTEPS": 50_000_000,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "SEED": 42,
        "SURVIVAL_REWARD": 0.01,
    }

    # Build env to get pure JAX functions
    env = SlimeVolleyEnv(config={"backend": "jax"})
    env.reset(seed=config["SEED"])

    step_fn = env.jax_step
    reset_fn = env.jax_reset
    n_agents = 2
    n_actions = 6  # Discrete(6)
    obs_dim = 12

    print(f"Training IPPO on SlimeVolley: {n_agents} agents, {n_actions} actions, obs_dim={obs_dim}")
    num_updates = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    print(f"  {config['NUM_ENVS']} envs, {num_updates} updates, {config['TOTAL_TIMESTEPS']:.0f} total steps")

    train_fn = jax.jit(make_train(config, step_fn, reset_fn, n_agents, n_actions, obs_dim))

    print("Compiling + training...")
    out = train_fn(jax.random.key(config["SEED"]))

    # ---- Summarize results ----
    ep_returns = np.array(out["metrics"]["returned_episode_returns"])
    ep_lengths = np.array(out["metrics"]["returned_episode_lengths"])
    ep_dones = np.array(out["metrics"]["returned_episode"])

    completed_returns = ep_returns[ep_dones > 0]
    completed_lengths = ep_lengths[ep_dones > 0]
    total_episodes = len(completed_returns)

    print(f"\nDone! {total_episodes} episodes completed")
    tail = max(1, total_episodes // 10)
    print(f"Right agent return (last {tail} eps): {completed_returns[-tail:].mean():.3f}")
    print(f"Right agent return (first {tail} eps): {completed_returns[:tail].mean():.3f}")
    print(f"Episode length (last {tail} eps): {completed_lengths[-tail:].mean():.1f}")
    print(f"Episode length (first {tail} eps): {completed_lengths[:tail].mean():.1f}")

    # ---- Plot learning curves ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_steps_cfg = config["NUM_STEPS"]
    num_envs_cfg = config["NUM_ENVS"]

    all_steps = []
    all_returns = []
    all_lengths = []
    for u in range(ep_returns.shape[0]):
        for s in range(ep_returns.shape[1]):
            env_step = (u * num_steps_cfg + s + 1) * num_envs_cfg
            for e in range(ep_returns.shape[2]):
                if ep_dones[u, s, e] > 0:
                    all_steps.append(env_step)
                    all_returns.append(ep_returns[u, s, e])
                    all_lengths.append(ep_lengths[u, s, e])

    all_steps = np.array(all_steps)
    all_returns = np.array(all_returns)
    all_lengths = np.array(all_lengths, dtype=np.float32)

    # Rolling mean
    window = max(1, len(all_returns) // 50)
    smoothed_returns = np.convolve(all_returns, np.ones(window) / window, mode="valid")
    smoothed_lengths = np.convolve(all_lengths, np.ones(window) / window, mode="valid")
    smoothed_steps = all_steps[window - 1:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(smoothed_steps, smoothed_returns, linewidth=1.5, color="#2563eb")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Right Agent Episode Return")
    ax1.set_title("Shared-Parameter IPPO Self-Play on Slime Volleyball")
    ax1.grid(True, alpha=0.3)

    ax2.plot(smoothed_steps, smoothed_lengths, linewidth=1.5, color="#dc2626")
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Episode Length")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("examples/slimevolley_training.png", dpi=150)
    print(f"Saved learning curve to examples/slimevolley_training.png")

    # ---- Export trained policy to ONNX ----
    params = out["runner_state"][0].params
    export_to_onnx(params, obs_dim, n_actions, activation=config["ACTIVATION"])

    # ---- Visualize trained policy as a GIF ----
    network = ActorCritic(n_actions, activation=config["ACTIVATION"])
    visualize_policy(network, params, gif_path="examples/episode.gif")
