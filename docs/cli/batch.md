# Using the Command Line Interface for Batch Jobs

The instructor CLI provides functionalities for managing batch jobs on both OpenAI and Anthropic platforms. This dual support allows users to leverage the strengths of both providers for their batch processing needs.

## Supported Providers

- **OpenAI**: Utilizes OpenAI's robust batch processing capabilities.
- **Anthropic**: Leverages Anthropic's advanced language models for batch operations.

To switch between providers, use the `--use-anthropic` flag in the relevant commands.

```bash
$ instructor batch --help

 Usage: instructor batch [OPTIONS] COMMAND [ARGS]...

 Manage OpenAI and Anthropic Batch jobs

╭─ Options ───────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                         │
╰─────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────╮
│ cancel             Cancel a batch job                                               │
│ create-from-file   Create a batch job from a file                                   │
│ download-file      Download the file associated with a batch job                    │
│ list               See all existing batch jobs                                      │
╰─────────────────────────────────────────────────────────────────────────────────────╯
```

## Creating a Batch Job

### View Jobs

```bash
$ instructor batch list --help

 Usage: instructor batch list [OPTIONS]

 See all existing batch jobs

╭─ Options ───────────────────────────────────────────────────────────────────────────╮
│ --limit                    INTEGER  Total number of batch jobs to show              │
│                                     [default: 10]                                   │
│ --poll                     INTEGER  Time in seconds to wait for the batch job to    │
│                                     complete                                        │
│                                     [default: 10]                                   │
│ --screen    --no-screen             Enable or disable screen output                 │
│                                     [default: no-screen]                            │
│ --use-anthropic                     Use Anthropic API instead of OpenAI             │
│                                     [default: False]                                │
│ --help                              Show this message and exit.                     │
╰─────────────────────────────────────────────────────────────────────────────────────╯
```

This returns a list of jobs as seen below:

```bash
$ instructor batch list --limit 5

                                   OpenAI Batch Jobs
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┓
┃ Batch ID             ┃ Created At          ┃ Status    ┃ Failed ┃ Completed ┃ Total ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━┩
│ batch_BSMSiMMy8on2D… │ 2024-06-19 15:10:21 │ cancelled │ 0      │ 298       │ 300   │
│ batch_pD5dqHmqjWYF5… │ 2024-06-19 15:09:38 │ completed │ 0      │ 15        │ 15    │
│ batch_zsTSsWVLgpEan… │ 2024-06-19 15:06:05 │ completed │ 0      │ 15        │ 15    │
│ batch_igaa2j9VBVw2Z… │ 2024-06-19 15:01:59 │ completed │ 0      │ 300       │ 300   │
│ batch_HcjI2wG46Y1LY… │ 2024-06-12 15:45:37 │ completed │ 0      │ 3         │ 3     │
└──────────────────────┴─────────────────────┴───────────┴────────┴───────────┴───────┘
```

### Create From File

You'll need to supply a valid .jsonl file to create a Batch job. Here's how you can create one using Instructor:

```python
from instructor.batch import BatchJob
from pydantic import BaseModel, Field
from typing import Literal

class Classification(BaseModel):
    label: Literal["SPAM", "NOT_SPAM"] = Field(
        ..., description="Whether the email is spam or not"
    )

emails = [
    "Hello there I'm a Nigerian prince and I want to give you money",
    "Meeting with Thomas has been set at Friday next week",
    "Here are some weekly product updates from our marketing team",
]

messages = [
    [
        {
            "role": "system",
            "content": f"Classify the following email {email}",
        }
    ]
    for email in emails
]

import json

with open("output.jsonl", "w") as f:
    for line in BatchJob.create_from_messages(
        messages,
        model="gpt-3.5-turbo",
        response_model=Classification,
        max_tokens=100,
    ):
        f.write(json.dumps(line) + "\n")
```

You can then import the .jsonl file using the `instructor batch create-from-file` command:

```bash
$ instructor batch create-from-file --help

Usage: instructor batch create-from-file [OPTIONS]

 Create a batch job from a file

╭─ Options ───────────────────────────────────────────────────────────────────────────╮
│ *  --file-path        TEXT  File containing the batch job requests [default: None]  │
│                             [required]                                              │
│    --use-anthropic          Use Anthropic API instead of OpenAI                     │
│                             [default: False]                                        │
│    --help                   Show this message and exit.                             │
╰─────────────────────────────────────────────────────────────────────────────────────╯
```

Example usage:

```bash
$ instructor batch create-from-file --file-path output.jsonl
```

### Cancelling a Batch Job

You can cancel an outstanding batch job using the `cancel` command:

```bash
$ instructor batch cancel --help

 Usage: instructor batch cancel [OPTIONS]

 Cancel a batch job

╭─ Options ───────────────────────────────────────────────────────────────────────────╮
│ *  --batch-id        TEXT  Batch job ID to cancel [default: None] [required]        │
│    --use-anthropic        Use Anthropic API instead of OpenAI                       │
│                           [default: False]                                          │
│    --help                 Show this message and exit.                               │
╰─────────────────────────────────────────────────────────────────────────────────────╯
```

Example usage:

```bash
$ instructor batch cancel --batch-id batch_BSMSiMMy8on2D
```

### Downloading Batch Job Results

To download the results of a completed batch job:

```bash
$ instructor batch download-file --help

 Usage: instructor batch download-file [OPTIONS]

 Download the file associated with a batch job

╭─ Options ───────────────────────────────────────────────────────────────────────────╮
│ *  --batch-id           TEXT  Batch job ID to download [default: None] [required]   │
│ *  --download-file-path TEXT  Path to download file to [default: None] [required]   │
│    --use-anthropic           Use Anthropic API instead of OpenAI                    │
│                              [default: False]                                       │
│    --help                    Show this message and exit.                            │
╰─────────────────────────────────────────────────────────────────────────────────────╯
```

Example usage:

```bash
$ instructor batch download-file --batch-id batch_pD5dqHmqjWYF5 --download-file-path results.jsonl
```

This comprehensive set of commands allows you to manage batch jobs efficiently, whether you're using OpenAI or Anthropic as your provider.
