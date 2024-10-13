from rich.console import Console
from rich.table import Table
from rich.live import Live
import typer
import time
from typing import Any

app = typer.Typer()

console = Console()


def generate_table(batch_jobs: list[Any], use_anthropic: bool):
    table = Table(
        title="Anthropic Batch Jobs" if use_anthropic else "OpenAI Batch Jobs",
    )

    table.add_column("Batch ID", style="dim")
    table.add_column("Created At")
    table.add_column("Processing Status")
    if not use_anthropic:
        table.add_column("Failed")
        table.add_column("Completed")
        table.add_column("Total")

    for batch_job in batch_jobs:
        if use_anthropic:
            table.add_row(
                str(batch_job.id),
                str(batch_job.created_at),
                str(batch_job.processing_status),
            )
        else:
            table.add_row(
                str(batch_job.id),
                str(batch_job.created_at),
                str(batch_job.status),
                str(getattr(batch_job.request_counts, "failed", "N/A")),
                str(getattr(batch_job.request_counts, "completed", "N/A")),
                str(getattr(batch_job.request_counts, "total", "N/A")),
            )

    return table


def get_jobs(limit: int = 10, use_anthropic: bool = False):
    if use_anthropic:
        from anthropic import Anthropic

        client = Anthropic()
        # Fetch batch jobs from Anthropic
        response = client.beta.messages.batches.list(limit=limit)
        return response.data  # Adjust based on Anthropic's API response structure
    else:
        from openai import OpenAI

        client = OpenAI()
        return client.batches.list(limit=limit).data


@app.command(name="list", help="See all existing batch jobs")
def watch(
    limit: int = typer.Option(10, help="Total number of batch jobs to show"),
    poll: int = typer.Option(
        10, help="Time in seconds to wait for the batch job to complete"
    ),
    screen: bool = typer.Option(False, help="Enable or disable screen output"),
    use_anthropic: bool = typer.Option(
        False, help="Use Anthropic API instead of OpenAI"
    ),
):
    """
    Monitor the status of the most recent batch jobs
    """
    batch_jobs = get_jobs(limit, use_anthropic)
    table = generate_table(batch_jobs, use_anthropic)

    with Live(table, refresh_per_second=2, screen=screen) as live_table:
        while True:
            batch_jobs = get_jobs(limit, use_anthropic)
            table = generate_table(batch_jobs, use_anthropic)
            live_table.update(table)
            time.sleep(poll)


@app.command(
    help="Create a batch job from a file",
)
def create_from_file(
    file_path: str = typer.Option(help="File containing the batch job requests"),
    use_anthropic: bool = typer.Option(
        False, help="Use Anthropic API instead of OpenAI"
    ),
):
    if use_anthropic:
        from anthropic import Anthropic

        client = Anthropic()
        with console.status(
            "[bold green]Creating Anthropic batch job...", spinner="dots"
        ):
            with open(file_path) as file:
                requests = [eval(line) for line in file]

            batch = client.beta.messages.batches.create(requests=requests)
        console.print(f"Anthropic batch job created with ID: {batch.id}")
    else:
        from openai import OpenAI

        client = OpenAI()
        with console.status(
            f"[bold green] Uploading batch job file...", spinner="dots"
        ):
            batch_input_file = client.files.create(
                file=open(file_path, "rb"), purpose="batch"
            )

        batch_input_file_id = batch_input_file.id

        with console.status(
            f"[bold green] Creating batch job from ID {batch_input_file_id}",
            spinner="dots",
        ):
            client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": "testing job"},
            )

    watch(limit=5, poll=2, screen=False, use_anthropic=use_anthropic)


@app.command(help="Cancel a batch job")
def cancel(
    batch_id: str = typer.Option(help="Batch job ID to cancel"),
    use_anthropic: bool = typer.Option(
        False, help="Use Anthropic API instead of OpenAI"
    ),
):
    try:
        if use_anthropic:
            from anthropic import Anthropic

            client = Anthropic()
            client.beta.messages.batches.cancel(batch_id)
        else:
            from openai import OpenAI

            client = OpenAI()
            client.batches.cancel(batch_id)
        watch(limit=5, poll=2, screen=False, use_anthropic=use_anthropic)
        console.log(f"[bold red]Job {batch_id} cancelled successfully!")
    except Exception as e:
        console.log(f"[bold red]Error cancelling job {batch_id}: {e}")


@app.command(help="Download the file associated with a batch job")
def download_file(
    batch_id: str = typer.Option(help="Batch job ID to download"),
    download_file_path: str = typer.Option(help="Path to download file to"),
    use_anthropic: bool = typer.Option(
        False, help="Use Anthropic API instead of OpenAI"
    ),
):
    try:
        if use_anthropic:
            from anthropic import Anthropic

            client = Anthropic()
            batch = client.beta.messages.batches.retrieve(batch_id)
            if batch.processing_status != "ended":
                raise ValueError("Only completed Jobs can be downloaded")

            results_url = batch.results_url
            if not results_url:
                raise ValueError("Results URL not available")

            # Download from results_url and save to download_file_path
            # This part depends on how you want to handle the download
            console.log(f"[bold green]Results available at: {results_url}")
            console.log("[bold yellow]Implement download logic for Anthropic results")
        else:
            from openai import OpenAI

            client = OpenAI()
            batch = client.batches.retrieve(batch_id=batch_id)
            status = batch.status

            if status != "completed":
                raise ValueError("Only completed Jobs can be downloaded")

            file_id = batch.output_file_id

            assert file_id, f"Equivalent Output File not found for {batch_id}"
            file_response = client.files.content(file_id)

            with open(download_file_path, "w") as file:
                file.write(file_response.text)

    except Exception as e:
        console.log(f"[bold red]Error downloading file for {batch_id}: {e}")
