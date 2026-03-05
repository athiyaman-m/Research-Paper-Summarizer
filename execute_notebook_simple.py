import io
import json
import traceback
from contextlib import redirect_stdout
from pathlib import Path


def execute_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    g = {"__name__": "__main__"}

    exec_count = 1
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        buf = io.StringIO()
        outputs = []
        cell["execution_count"] = exec_count
        exec_count += 1

        try:
            with redirect_stdout(buf):
                exec(src, g, g)
            txt = buf.getvalue()
            if txt:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": txt,
                })
        except Exception as exc:
            txt = buf.getvalue()
            if txt:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": txt,
                })
            tb = traceback.format_exc()
            outputs.append({
                "output_type": "error",
                "ename": type(exc).__name__,
                "evalue": str(exc),
                "traceback": tb.splitlines(),
            })
            cell["outputs"] = outputs
            path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
            raise RuntimeError(f"Notebook execution failed at cell index {idx}: {exc}") from exc

        cell["outputs"] = outputs

    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")


if __name__ == "__main__":
    nb_path = Path("research_paper_novelty_experiments.ipynb").resolve()
    execute_notebook(nb_path)
    print(f"Executed notebook: {nb_path}")
