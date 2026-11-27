# Regimex documentation

| I want to… | Start here |
| --- | --- |
| Get access or understand what my role can do | [Sign-in and roles](access.md) |
| Analyze an export, compare two results or understand the chart | [Analysis workspace](usage.md) |
| Train a model, watch its progress or publish it | [Training workspace](training.md) |
| Generate an experiment or submit a job from a shell | [Command-line workflows](cli.md) |
| Call the service from a simulator | [Integration guide](integration.md) and [JSON export schema](../schemas/experiment-v2.schema.json) |
| Install or update the service | [Single-host deployment](deployment.md) or [multiple application nodes](multi-host.md) |
| Operate an installation | [Access and keys](access.md), [storage and recovery](storage.md), [load testing](load-testing.md) |
| Modify or package Regimex | [Development setup and checks](development.md) and [release packaging](releasing.md) |

## Reference

- [Architecture](architecture.md) traces a request from upload to report and maps its components to the code.
- [API workflows](api.md) covers authentication, endpoints, idempotency and job states. The running service exposes its interactive schema at `/docs`.
- [Data and model semantics](models.md) defines observations, availability, direction labels and the model bundle contract.
- [Model quality](quality.md) shows where to find scores for a saved model and how to interpret them.
- [Evaluation reports](evaluations/README.md) contains experiments, their data and the commands used to reproduce them.
