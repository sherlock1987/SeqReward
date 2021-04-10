#!/bin/bash
#1. seed, the saving dir(seed)
#1.1 write stuff to config file
#2. evaluation.py ts write down the result into the same file
time=$(date "+%Y-%m-%d--%H:%M:%S")
root=`pwd`
device=0
echo "${time}"
echo "${root}"

config_path=${root}/"convlab/spec/a2c.json"
for paramater in 10.0 50.0 100.0 150.0 200.0 300.0 400.0 500.0 600.0
do
  {
    # write in the config file and run this stuff.
    new_config_path=${root}/"convlab/spec/a2c_${paramater}.json"
    echo "${paramater}"
    echo "${new_config_path}"
    echo '
    {
  "rule_a2c": {
    "agent": [{
      "name": "DialogAgent",
      "dst": {
        "name": "RuleDST"
      },
      "state_encoder": {
        "name": "MultiWozStateEncoder"
      },
      "action_decoder": {
        "name": "MultiWozVocabActionDecoder"
      },
      "algorithm": {
        "name": "ActorCritic",
        "action_pdtype": "default",
        "action_policy": "default",
        "explore_var_spec": null,
        "gamma": 0.99,
        "lam": 0.95,
        "num_step_returns": null,
        "entropy_coef_spec": {
          "name": "linear_decay",
          "start_val": 0.001,
          "end_val": 0.001,
          "start_step": 1000,
          "end_step": 500000,
        },
        "policy_loss_coef": 1.0,
        "val_loss_coef": 1.0,
        "training_frequency": '${paramater}',
      },
      "memory": {
        "name": "OnPolicyReplay"
      },
      "net": {
        "type": "MLPNet",
        "shared": false,
        "hid_layers": [100],
        "hid_layers_activation": "relu",
        "clip_grad_val": 10.0,
        "use_same_optim": false,
        "actor_optim_spec": {
          "name": "Adam",
          "lr": 0.0001
        },
        "critic_optim_spec": {
          "name": "Adam",
          "lr": 0.0001
        },
        "lr_scheduler_spec": {
          "name": "StepLR",
          "step_size": 2000,
          "gamma": 0.999,
        },
        "gpu": false
      }
    }],
    "env": [{
      "name": "multiwoz",
      "action_dim": 300,
      "observation_dim": 392,
      "max_t": 40,
      "max_frame": 40000,
      "user_policy": {
        "name": "UserPolicyAgendaMultiWoz"
      },
      "sys_policy": {
        "name": "RuleBasedMultiwozBot"
      }
    }],
    "meta": {
      "distributed": false,
      "num_eval": 100,
      "eval_frequency": 1000,
      "max_tick_unit": "total_t",
      "max_trial": 1,
      "max_session": 1,
      "resources": {
        "num_cpus": 1,
        "num_gpus": 0
      }
    }
  }
}
'> ${new_config_path}
  # run python file
  CUDA_VISIBLE_DEVICES=${device} python run.py a2c_${paramater}.json rule_a2c train
  }&
done
wait
sleep $[10]