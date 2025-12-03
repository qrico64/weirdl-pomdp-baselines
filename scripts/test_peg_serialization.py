"""Test script to verify PegInsertionEnv serialization works correctly."""

import pickle
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.meta.mujoco.peg_insertion import PegInsertionEnv


def test_basic_serialization():
    """Test basic pickle serialization and deserialization."""
    print("=" * 60)
    print("TEST 1: Basic Pickle Serialization")
    print("=" * 60)

    # Create environment with specific configuration
    env = PegInsertionEnv(
        num_train_tasks=5,
        num_eval_tasks=3,
        max_episode_steps=100,
        task_mode="random_target",
        goal_conditioning="yes",
        goal_noise_magnitude=0.1,
        goal_noise_type="normal",
        seed=42
    )

    print(f"✓ Created original environment")
    print(f"  - num_train_tasks: {env.num_train_tasks}")
    print(f"  - num_eval_tasks: {env.num_eval_tasks}")
    print(f"  - task_mode: {env.task_mode}")
    print(f"  - goal_conditioning: {env.goal_conditioning}")
    print(f"  - seed: {env.seed}")

    # Pickle the environment
    try:
        pickled_env = pickle.dumps(env)
        print(f"✓ Successfully pickled environment ({len(pickled_env)} bytes)")
    except Exception as e:
        print(f"✗ FAILED to pickle environment: {e}")
        return False

    # Unpickle the environment
    try:
        restored_env = pickle.loads(pickled_env)
        print(f"✓ Successfully unpickled environment")
    except Exception as e:
        print(f"✗ FAILED to unpickle environment: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify attributes are preserved
    attrs_to_check = [
        'num_train_tasks', 'num_eval_tasks', 'task_mode',
        'goal_conditioning', 'goal_noise_magnitude', 'seed',
        '_max_episode_steps', 'goal_noise_type'
    ]

    all_match = True
    for attr in attrs_to_check:
        original_val = getattr(env, attr)
        restored_val = getattr(restored_env, attr)
        if original_val != restored_val:
            print(f"✗ Attribute '{attr}' mismatch: {original_val} vs {restored_val}")
            all_match = False
        else:
            print(f"  ✓ {attr}: {original_val}")

    if all_match:
        print("✓ All attributes preserved correctly!")

    return all_match


def test_functional_serialization():
    """Test that the environment works after deserialization."""
    print("\n" + "=" * 60)
    print("TEST 2: Functional Test After Deserialization")
    print("=" * 60)

    # Create and pickle environment (task_mode="fixed" requires both to be 1)
    env = PegInsertionEnv(
        num_train_tasks=1,
        num_eval_tasks=1,
        task_mode="fixed",
        seed=123
    )

    print(f"✓ Created original environment")

    # Reset to a specific task
    env.reset_task(0)
    obs1 = env.reset()
    print(f"✓ Reset original environment, obs shape: {obs1.shape}")

    # Take a step
    action = env.action_space.sample()
    obs2, reward, done, info = env.step(action)
    print(f"✓ Took step in original environment")
    print(f"  - action shape: {action.shape}")
    print(f"  - obs shape: {obs2.shape}")
    print(f"  - reward: {reward:.4f}")

    # Pickle and unpickle
    pickled_env = pickle.dumps(env)
    restored_env = pickle.loads(pickled_env)
    print(f"✓ Pickled and restored environment")

    # Test restored environment works
    try:
        restored_env.reset_task(0)
        obs3 = restored_env.reset()
        print(f"✓ Reset restored environment, obs shape: {obs3.shape}")

        action2 = restored_env.action_space.sample()
        obs4, reward2, done2, info2 = restored_env.step(action2)
        print(f"✓ Took step in restored environment")
        print(f"  - action shape: {action2.shape}")
        print(f"  - obs shape: {obs4.shape}")
        print(f"  - reward: {reward2:.4f}")

        print("✓ Restored environment is fully functional!")
        return True

    except Exception as e:
        print(f"✗ FAILED to use restored environment: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiprocessing_serialization():
    """Test serialization with multiprocessing (the main use case)."""
    print("\n" + "=" * 60)
    print("TEST 3: Multiprocessing Compatibility")
    print("=" * 60)

    import multiprocessing as mp

    def worker(env):
        """Worker function that uses the environment."""
        env.reset_task(0)
        obs = env.reset()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        return reward

    try:
        # Create environment
        env = PegInsertionEnv(
            num_train_tasks=1,
            num_eval_tasks=1,
            task_mode="fixed",
            seed=999
        )
        print(f"✓ Created environment")

        # Try to pickle for multiprocessing
        ctx = mp.get_context('spawn')  # Use spawn to ensure clean serialization
        with ctx.Pool(processes=1) as pool:
            result = pool.apply(worker, (env,))
            print(f"✓ Successfully used environment in subprocess")
            print(f"  - Reward from subprocess: {result:.4f}")

        return True

    except Exception as e:
        print(f"✗ FAILED multiprocessing test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PegInsertionEnv Serialization Test Suite")
    print("=" * 60 + "\n")

    results = {
        "Basic Serialization": test_basic_serialization(),
        "Functional Test": test_functional_serialization(),
        "Multiprocessing": test_multiprocessing_serialization()
    }

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All tests passed! PegInsertionEnv is fully serializable.")
        return 0
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())
