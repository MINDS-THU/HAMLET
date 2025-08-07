import os
import sys
import argparse
import unittest

def run_tests(target: str = None):
    base_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(base_dir, ".."))

    # Add the project root to sys.path so 'general_tools' can be imported
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    test_dir = os.path.join(base_dir, target) if target else base_dir

    if not os.path.exists(test_dir):
        print(f"Error: No such test directory: {test_dir}")
        exit(1)

    print(f"Running tests in: {test_dir}")
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run unit tests for default_tools.")
    parser.add_argument(
        "--tool",
        type=str,
        default=None,
        help="Optional. Name of the tool subfolder to test (e.g. 'kb_repo_management'). If omitted, runs all tests."
    )
    args = parser.parse_args()
    run_tests(args.tool)

if __name__ == "__main__":
    main()
