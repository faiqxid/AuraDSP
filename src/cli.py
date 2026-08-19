import sys
import click
import json
import dataclasses
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add src to python path dynamically if run directly
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from detector import AIMusicDetector
from remaster import RemasterPipeline

console = Console()


def print_detection_results(filename: str, res) -> None:
    """Print detector result table."""
    verdict_colors = {
        "HUMAN": "bold green",
        "LIKELY_HUMAN": "green",
        "UNCERTAIN": "yellow",
        "LIKELY_AI": "red",
        "AI_GENERATED": "bold red",
    }
    v_color = verdict_colors.get(res.verdict, "white")

    console.print()
    console.print(Panel(
        f"File: [bold]{filename}[/bold]\n"
        f"Verdict: [{v_color}]{res.verdict}[/{v_color}] (Confidence: {res.confidence})\n"
        f"AI Probability: [bold]{res.percentage}%[/bold]",
        title="Detection Results",
        border_style="magenta" if "AI" in res.verdict else "green" if "HUMAN" in res.verdict else "yellow"
    ))

    table = Table(title="Analysis Vectors", border_style="dim")
    table.add_column("Vector", style="cyan")
    table.add_column("AI Score", justify="right")
    table.add_column("Verdict", justify="center")

    vector_names = {
        "spectral": "Spectral Fingerprint (Vocoder peaks)",
        "timing": "Timing Precision (Onset grid)",
        "phase": "Phase Coherence (Entropy)",
        "noise_floor": "Noise Floor (Ambient statistics)",
        "metadata": "Metadata & Watermarks (C2PA/Tags)",
    }

    for k, score in res.vector_scores.items():
        v_name = vector_names.get(k, k)
        v_verdict = res.vector_verdicts.get(k, "uncertain")
        v_v_color = "red" if v_verdict == "ai" else "green" if v_verdict == "human" else "yellow"
        table.add_row(
            v_name,
            f"{round(score * 100, 1)}%",
            f"[{v_v_color}]{v_verdict.upper()}[/{v_v_color}]"
        )

    console.print(table)

    console.print("\n[bold]Diagnostic Notes:[/bold]")
    for note in res.notes:
        console.print(f" • [dim]{note}[/dim]")
    console.print()


@click.group()
def main():
    """AI Music Detector & Remaster CLI — detect and remaster AI-generated music."""
    pass


@main.command("detect")
@click.argument("filepath", type=click.Path(exists=True, dir_okay=False))
@click.option("--json-out", "json_output", is_flag=True, help="Output results in JSON format.")
@click.option("--max-seconds", default=180.0, type=float, help="Max audio duration to analyze (default: 180s).")
@click.option("--sr", "sample_rate", default=22050, type=int, help="Sample rate to use (default: 22050Hz).")
def detect(filepath: str, json_output: bool, max_seconds: float, sample_rate: int):
    """Detect AI-generated audio in a single file."""
    if json_output:
        try:
            detector = AIMusicDetector(sample_rate=sample_rate, max_seconds=max_seconds)
            res = detector.detect(filepath)
            print(json.dumps(dataclasses.asdict(res), indent=2))
            sys.exit(0)
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.exit(1)

    console.print(Panel.fit(
        "[bold magenta]🎵 AI Music Detector v0.1.0[/bold magenta]\n[dim]Multi-vector heuristic scanner[/dim]",
        border_style="magenta"
    ))

    p = Path(filepath)
    console.print(f"[bold cyan]Input file:[/bold cyan] {p.name}")

    detector = AIMusicDetector(sample_rate=sample_rate, max_seconds=max_seconds)

    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[yellow]Analyzing audio features...[/yellow]", total=None)
        try:
            res = detector.detect(filepath)
            progress.update(task, completed=True)
        except Exception as e:
            progress.update(task, completed=True)
            console.print(f"[bold red]Error during analysis:[/bold red] {e}")
            sys.exit(1)

    print_detection_results(p.name, res)


@main.command("remaster")
@click.argument("folder_path", type=click.Path(exists=True, file_okay=False))
@click.option("--intensity", default="aggressive", type=click.Choice(["quality", "aggressive"]),
              help="Remaster intensity mode.")
@click.option("--reference", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Reference WAV/MP3 file to match EQ/Dynamics/LUFS.")
def remaster(folder_path: str, intensity: str, reference: str | None):
    """Batch remaster all audio files in a folder into folder/output/*.wav."""
    folder = Path(folder_path)
    console.print(Panel.fit(
        f"[bold magenta]🎀 Batch Remaster[/bold magenta]\n[dim]{folder}[/dim]",
        border_style="magenta"
    ))

    pipeline = RemasterPipeline(
        output_dir=folder / "output",
        sample_rate=44100,
        use_cuda=True,
        intensity=intensity,
    )
    detector = AIMusicDetector(sample_rate=22050, max_seconds=180.0)

    try:
        results = pipeline.batch_remaster(folder, reference_path=reference)
    except Exception as e:
        console.print(f"[bold red]Batch failed:[/bold red] {e}")
        sys.exit(1)

    table = Table(title="Batch Remaster Results", border_style="magenta")
    table.add_column("File", style="cyan")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_column("Status", justify="center")

    for result in results:
        input_name = Path(result.original_path).name
        if not result.filepath:
            err = "; ".join(result.notes or ["Unknown error"])
            table.add_row(input_name, "-", "-", f"[red]FAIL[/red]")
            continue

        try:
            before = detector.detect(result.original_path)
            after = detector.detect(result.filepath)
            status = "[green]PASS[/green]" if after.percentage < 5 else "[yellow]CHECK[/yellow]"
            table.add_row(
                input_name,
                f"{before.percentage:.2f}%",
                f"{after.percentage:.2f}%",
                status,
            )
        except Exception as e:
            table.add_row(input_name, "-", "-", f"[yellow]DONE[/yellow]")

    console.print(table)
    console.print(f"\n[bold green]Output folder:[/bold green] {folder / 'output'}")


if __name__ == "__main__":
    main()