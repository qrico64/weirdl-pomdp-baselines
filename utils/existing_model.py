from ruamel.yaml import YAML
from policies.learner import Learner
from utils import system
from pathlib import Path

def pull_model(config_file: str, checkpoint_num: str, override_args: dict):
    assert config_file.startswith("variant_"), f"Unable to load incorrect name: {config_file}"
    yaml = YAML()
    print(f"Pulling from {config_file} :")
    v = yaml.load(open(config_file))
    seed = v["seed"]
    system.reproduce(seed)
    learner = Learner(
        env_args=v["env"] | override_args.get("env", {}),
        train_args=v["train"] | override_args.get("train", {}),
        eval_args=v["eval"] | override_args.get("eval", {}),
        policy_args=v["policy"] | override_args.get("policy", {}),
        seed=seed,
    )
    
    agent_files = [fp for fp in (Path(config_file).parent / "save").iterdir() if fp.is_file() and fp.name.endswith("agent_")]
    if checkpoint_num == "latest":
        last_agent_file = max(agent_files, key=lambda fp: int(fp.name.split('_')[1]))
    elif checkpoint_num == "bestperf":
        last_agent_file = max(agent_files, key=lambda fp: float(fp.name.split('perf')[1][:-3]))
    else:
        agent_files_exact = [fp for fp in agent_files if fp.name.startswith(f"agent_{checkpoint_num}_")]
        assert len(agent_files_exact) > 0, f"{agent_files_exact}"
        last_agent_file = agent_files_exact[0]
    
    learner.load_ingeneral(last_agent_file)

    return learner
