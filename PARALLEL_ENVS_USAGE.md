# Parallel Environment Collection - Usage Guide

This document explains how to use the parallel environment collection system to speed up rollout collection in the meta-learning training pipeline.

## Overview

The system provides true parallelization of environment rollouts by:
1. Running multiple environment instances in separate processes
2. Each environment can be reset with a different task
3. All environments compute steps simultaneously (true parallelization)
4. Compatible with both Markov and Memory-based agents

## Components

### 1. ParallelEnvManager (`envs/parallel_env_manager.py`)
A multiprocessing-based environment manager that:
- Spawns worker processes, each running one environment
- Communicates via pipes for non-blocking command sending
- Supports per-environment task-based resets

### 2. Helper Function (`envs/meta/make_env.py`)
```python
make_parallel_env(env_id, episodes_per_task, num_workers, seed=None, oracle=False, **kwargs)
```

### 3. Parallel Collection Method (`policies/learner.py`)
```python
learner.collect_rollouts_parallel(num_rollouts, random_actions=False, parallel_env=None)
```

## Usage Example

### Basic Usage in Learner

```python
from envs.meta.make_env import make_env, make_parallel_env

# In learner.py init_env method:
class Learner:
    def init_env(self, env_type, env_name, max_rollouts_per_task,
                 num_train_tasks, num_eval_tasks, num_parallel_workers=4, **kwargs):

        # Create single env for evaluation
        self.eval_env = make_env(
            env_name,
            max_rollouts_per_task,
            seed=self.seed + 1,
            num_train_tasks=num_train_tasks,
            num_eval_tasks=num_eval_tasks,
            **kwargs,
        )

        # Create parallel envs for training
        if num_parallel_workers > 1:
            self.train_env_parallel = make_parallel_env(
                env_name,
                max_rollouts_per_task,
                num_workers=num_parallel_workers,
                seed=self.seed,
                num_train_tasks=num_train_tasks,
                num_eval_tasks=num_eval_tasks,
                **kwargs,
            )
        else:
            self.train_env_parallel = None

        # Also keep single env for compatibility
        self.train_env = make_env(
            env_name,
            max_rollouts_per_task,
            seed=self.seed,
            num_train_tasks=num_train_tasks,
            num_eval_tasks=num_eval_tasks,
            **kwargs,
        )
```

### Using Parallel Collection

```python
# In learner.py train method:
def train(self):
    self._start_training()

    # Initial pool collection
    if self.num_init_rollouts_pool > 0:
        logger.log("Collecting initial pool of data..")
        while self._n_env_steps_total < self.num_init_rollouts_pool * self.max_trajectory_len:
            # Use parallel collection if available
            self.collect_rollouts_parallel(
                num_rollouts=1,
                random_actions=True,
                parallel_env=self.train_env_parallel,  # Will fallback to sequential if None
            )

    # Main training loop
    while self._n_env_steps_total < self.n_env_steps_total:
        # Collect data in parallel
        env_steps = self.collect_rollouts_parallel(
            num_rollouts=self.num_rollouts_per_iter,
            parallel_env=self.train_env_parallel,
        )

        # Update policy
        train_stats = self.update(...)
        self.log_train_stats(train_stats)
```

## How Parallelization Works

### Phase-by-Phase Execution

1. **Action Computation** (Sequential, on GPU/CPU)
   ```python
   for worker_id in active_workers:
       action = self.agent.act(obs[worker_id])  # Policy forward pass
   ```

2. **Environment Steps** (Parallel, in worker processes)
   ```python
   # Send commands to ALL workers (non-blocking)
   for worker_id in active_workers:
       remote.send(('step', action))  # Returns immediately

   # Workers compute env.step() simultaneously in their processes
   ```

3. **Result Collection** (Blocking)
   ```python
   # Collect results after all workers finish
   for worker_id in active_workers:
       result = remote.recv()  # Blocks until this worker is done
   ```

4. **Buffer Updates** (Sequential)
   ```python
   for worker_id, result in results:
       self.policy_storage.add_sample(...)  # Add to replay buffer
   ```

### Key Insight
The speedup comes from **Phase 2**: all workers compute `env.step()` in parallel while the main process waits. Even though we collect results sequentially, the actual environment computation happens simultaneously.

## Performance Considerations

### When to Use Parallel Collection
- **Heavy environment computation**: MuJoCo, complex physics simulations
- **Multiple rollouts needed**: num_rollouts >> 1
- **CPU-bound environments**: Parallel processes utilize multiple cores

### When NOT to Use
- **Simple environments**: Overhead of multiprocessing may outweigh benefits
- **GPU-heavy policy**: If policy evaluation is the bottleneck, parallelizing envs won't help much
- **Single rollout**: No benefit if num_rollouts = 1

### Optimal Number of Workers
- Start with `num_workers = num_cpu_cores - 2` (leave some for main process and system)
- For MuJoCo environments: 4-8 workers is typically good
- Monitor CPU utilization to tune

## Configuration Example

```yaml
# In your config YAML
env:
  env_type: meta
  env_name: AntDir-v0
  max_rollouts_per_task: 2
  num_train_tasks: 50
  num_eval_tasks: 20
  num_parallel_workers: 6  # NEW: Number of parallel environment workers

train:
  num_rollouts_per_iter: 12  # Will be distributed across 6 workers (2 each)
  ...
```

## Task Assignment

Each worker can be reset with a different task:

```python
# In collect_rollouts_parallel:
for worker_id in range(num_workers):
    task = self.train_tasks[np.random.randint(len(self.train_tasks))]
    parallel_env.remotes[worker_id].send(('reset', {'task': task}))
```

This allows:
- Diverse task sampling across workers
- Independent rollout collection
- Each worker finishes at its own pace

## Cleanup

The `ParallelEnvManager` automatically cleans up worker processes:

```python
# Automatic cleanup on deletion
del parallel_env

# Or explicit cleanup
parallel_env.close()
```

## Troubleshooting

### Issue: Workers hanging/not responding
- **Cause**: Deadlock in pipe communication
- **Solution**: Ensure all `send()` calls have matching `recv()` calls

### Issue: Lower than expected speedup
- **Cause**: Policy evaluation is the bottleneck, not environment
- **Solution**: Profile your code to identify actual bottleneck

### Issue: "Too many open files" error
- **Cause**: Each worker creates file descriptors for pipes
- **Solution**: Reduce `num_workers` or increase system limits (`ulimit -n`)

### Issue: Import errors in worker processes
- **Cause**: Worker processes need to import environment modules
- **Solution**: Ensure all environment dependencies are properly installed and importable

## Compatibility

- ✅ Compatible with Markov agents (AGENT_ARCHS.Markov)
- ✅ Compatible with Memory agents (AGENT_ARCHS.Memory)
- ✅ Compatible with Memory-Markov agents (AGENT_ARCHS.Memory_Markov)
- ✅ Works with VariBadWrapper
- ✅ Handles task-based resets for meta-learning
- ✅ Falls back to sequential collection if parallel_env=None
