from slime_volleyball import slimevolley_env
import ray
from ray.rllib.algorithms import ppo
from ray.rllib.policy import policy
from ray.rllib.examples.policy import random_policy
from ray import train, tune
from ray.tune import CLIReporter
from ray.air.integrations import wandb


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "agent0"


def eval_policy_mapping_fn(agent_id, episode, workers, **kwargs):
    return "agent0" if agent_id == "agent_right" else "random"


if __name__ == "__main__":
    ray.init(num_cpus=4, num_gpus=0, local_mode=False)
    alg_config = (
        ppo.PPOConfig()
        .environment(env=slimevolley_env.SlimeVolleyEnv, disable_env_checking=False)
        .multi_agent(
            policies={
                "agent0": policy.PolicySpec(
                    policy_class=None,
                    observation_space=None,
                    action_space=None,
                    config={"gamma": 0.99},
                ),
                "random": policy.PolicySpec(policy_class=random_policy.RandomPolicy),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["agent0"],
        )
        .evaluation(
            evaluation_config={
                "multiagent": {"policy_mapping_fn": eval_policy_mapping_fn}
            }
        )
    )

    tuner = tune.Tuner(
        "PPO",
        run_config=train.RunConfig(
            stop={"episode_reward_mean": 150},
            callbacks=[
                wandb.WandbLoggerCallback(project="slime_volleyball", group="testing")
            ],
        ),
        param_space=alg_config,
    )

    tuner.fit()
