---
title: Managing Batch Jobs with OpenAI CLI
description: Learn how to create, list, and cancel batch jobs using the OpenAI Command Line Interface (CLI) for efficient job management.
---

# Using the Command Line Interface

The instructor CLI provides functionalities for managing batch jobs on OpenAI

```bash
$ instructor batch --help

 Usage: instructor batch [OPTIONS] COMMAND [ARGS]...

 Manage OpenAI Batch jobs

╭─ Options ───────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                         │
╰─────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ──────────────────────────────────────────────────────────────────────────╮
│ cancel             Cancel a batch job                                               │
│ create-from-file   Create a batch job from a file                                   │
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
│ --help                              Show this message and exit.                     │
╰─────────────────────────────────────────────────────────────────────────────────────╯
```

This returns a list of jobs as seen below

```
$ instructor batch list --limit 9

                                   OpenAI Batch Jobs
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┓
┃ Batch ID             ┃ Created At          ┃ Status    ┃ Failed ┃ Completed ┃ Total ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━┩
│ batch_BSMSiMMy8on2D… │ 2024-06-19 15:10:21 │ cancelled │ 0      │ 298       │ 300   │
│ batch_pD5dqHmqjWYF5… │ 2024-06-19 15:09:38 │ completed │ 0      │ 15        │ 15    │
│ batch_zsTSsWVLgpEan… │ 2024-06-19 15:06:05 │ completed │ 0      │ 15        │ 15    │
│ batch_igaa2j9VBVw2Z… │ 2024-06-19 15:01:59 │ completed │ 0      │ 300       │ 300   │
│ batch_HcjI2wG46Y1LY… │ 2024-06-12 15:45:37 │ completed │ 0      │ 3         │ 3     │
│ batch_YiRKLAmKBhwxM… │ 2024-06-12 15:09:44 │ completed │ 0      │ 3         │ 3     │
│ batch_hS0XGlXzTVS7S… │ 2024-06-12 15:05:59 │ completed │ 0      │ 3         │ 3     │
│ batch_6s4FmcaV7woam… │ 2024-06-12 14:26:34 │ completed │ 0      │ 3         │ 3     │
└──────────────────────┴─────────────────────┴───────────┴────────┴───────────┴───────┘
```

### Create From File

You'll need to supply a valid .jsonl file in order to be able to create a Batch job.

??? info "Don't have a `.jsonl` file on hand?"

    You can use Instructor to create the `.jsonl` with nothing more than simple pydantic and our `BatchJob` object as seen below.

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
                "role": "user",
                "content": f"Classify the following email {email}",
            }
        ]
        for email in emails
    ]

    BatchJob.create_from_messages(
        messages,
        model="gpt-3.5-turbo",
        response_model=Classification,
        max_tokens=100,
        file_path="output.jsonl"
    )
    ```

You can then import in the .jsonl file using the `instructor batch create-from-file` command

```bash
$ instructor batch create-from-file --help

Usage: instructor batch create-from-file [OPTIONS]

 Create a batch job from a file

╭─ Options ───────────────────────────────────────────────────────────────────────────╮
│ *  --file-path        TEXT  File containing the batch job requests [default: None]  │
│                             [required]                                              │
│    --help                   Show this message and exit.                             │
╰─────────────────────────────────────────────────────────────────────────────────────╯
```

### Cancelling a Batch Job

You can also cancel an outstanding batch job by using the `cancel` command.

```bash
$ instructor batch cancel --help

 Usage: instructor batch cancel [OPTIONS]

 Cancel a batch job

╭─ Options ───────────────────────────────────────────────────────────────────────────╮
│ *  --batch-id        TEXT  Batch job ID to cancel [default: None] [required]        │
│    --help                  Show this message and exit.                              │
╰─────────────────────────────────────────────────────────────────────────────────────╯
```
