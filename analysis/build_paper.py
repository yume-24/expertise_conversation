"""Build the working LaTeX manuscript, keeping all build products in outputs."""
from pathlib import Path
import os
import shutil
import subprocess
from .config import ROOT


def main():
    for tool in ['pdflatex', 'bibtex']:
        if shutil.which(tool) is None:
            raise SystemExit(f'{tool} is required. Install a TeX distribution; Python dependencies do not include TeX.')
    source = ROOT / 'paper/mainPaper2.tex'
    target = ROOT / 'outputs/paper/manuscript'
    target.mkdir(parents=True, exist_ok=True)
    command = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', f'-output-directory={target}', source.name]
    for step in range(3):
        process = subprocess.run(command, cwd=source.parent, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (target / 'compile_stdout.txt').write_text(process.stdout)
        if process.returncode:
            raise SystemExit(f'LaTeX failed; see {target / "compile_stdout.txt"}')
        if step == 0:
            env = os.environ.copy()
            env['BIBINPUTS'] = str(source.parent) + os.pathsep + env.get('BIBINPUTS', '')
            process = subprocess.run(['bibtex', source.stem], cwd=target, env=env,
                                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (target / 'bibtex_stdout.txt').write_text(process.stdout)
            if process.returncode:
                raise SystemExit(f'BibTeX failed; see {target / "bibtex_stdout.txt"}')
    print(target / (source.stem + '.pdf'))


if __name__ == '__main__':
    main()
