"""Execute the two paper notebooks in order, each in a fresh Jupyter kernel."""
from pathlib import Path
import argparse
import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=['metrics', 'statistics'], help='Run only one stage.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    stages = [('metrics', '01_static_geometric_metrics.ipynb'),
              ('statistics', '02_statistical_significance.ipynb')]
    for stage, name in stages:
        if args.only and args.only != stage:
            continue
        path = root / 'notebooks/paper' / name
        notebook = nbformat.read(path, as_version=4)
        print(f'Executing {path.relative_to(root)}', flush=True)
        # Bind the kernel to this interpreter, even if the system kernel differs.
        import sys
        client = NotebookClient(notebook, kernel_name='python3', timeout=1800,
                                resources={'metadata': {'path': str(path.parent)}})
        manager = client.create_kernel_manager()
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        try:
            client.execute()
        finally:
            nbformat.write(notebook, path)
        print(f'Completed {stage}', flush=True)


if __name__ == '__main__':
    main()
