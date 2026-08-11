# Foldways API

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=white)](https://www.python.org)
[![Modal](https://img.shields.io/badge/deployed%20on-Modal-7C3AED.svg?logo=modal&logoColor=white)](https://modal.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![uv](https://img.shields.io/badge/uv-DE5FE9.svg?logo=uv&logoColor=white)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/badge/Ruff-000000.svg?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)

Deploy popular protein engineering tools behind a REST API in your Modal
account.

## Contents

- [About Foldways API](#about-foldways-api)
- [How it works](#how-it-works)
- [Services](#services)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Usage with an agent](#usage-with-an-agent)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## About Foldways API

Computational tools for protein engineering have become incredibly powerful over
the past few years. An array of models now covers structure prediction, inverse
folding, binder design, and stability prediction. Altogether, they have
transformed the field.

However, structure prediction and protein design tools each ship their own
dependencies, weights, installation methods, CLI, I/O formats, and GPU
assumptions. Running several of them has significant operational overhead.

Foldways API puts them behind one REST API with a shared job model. This makes
it possible to submit a workload, poll for status, and download results the same
way for every service. The interface is uniform and self-describing, so both
people and AI agents can use it.

It runs on Modal so GPU-intensive workloads stay cost-efficient. The API is not
a hosted service, so you deploy it to your own Modal account and control the
resources, usage, and billing.

## How it works

Foldways API runs on [Modal](https://modal.com), which is a platform for running
Python on cloud GPUs without managing servers. The API is deployed as a single
Modal App, and Modal builds a container image for each service. Containers start
when a workload arrives, and Modal scales them to zero when the workload is
done. You are billed for the seconds a GPU is actually busy. Each service
declares its own GPU type, timeout, and concurrency limit.

Every tool is exposed through the same lifecycle:

1. **Submit.** `POST /jobs` or `POST /batches` with a service name and its
   params returns ids immediately.
2. **Poll.** `GET /jobs/{job_id}` or `GET /batches/{batch_id}` reports status.
3. **Download.** `GET /jobs/{job_id}/download` returns the tool's entire output
   directory as a zip.

Foldways API creates a Modal volume in your account. The volume is used to store
ML weights and any large artifacts needed to deploy the protein engineering
tools. It is also used to store workload results, so they outlive the container
that produced them and stay available after the GPU has scaled down.

Each service publishes its parameter schema at `GET /services/{name}`, and the
whole API is browsable as interactive API docs after deployment to Modal.

Boltz-2 and Chai use multiple sequence alignments (MSAs). Foldways API builds
them with the [ColabFold MSA server](https://github.com/sokrypton/ColabFold), so
each protein sequence you submit is sent to that third-party service. This is
done to avoid storing large MSA files in the Modal volume.

## Services

Supported and planned services.

| Service       | Status       | Codebase                                                                | Type                 |
| ------------- | ------------ | ----------------------------------------------------------------------- | -------------------- |
| Boltz-2       | ✅ Supported | [jwohlwend/boltz](https://github.com/jwohlwend/boltz)                   | Structure prediction |
| BoltzGen      | ✅ Supported | [HannesStark/boltzgen](https://github.com/HannesStark/boltzgen)         | De novo design       |
| ESMC          | ✅ Supported | [Biohub/esm](https://github.com/Biohub/esm)                             | Representation       |
| ESMFold2      | ✅ Supported | [Biohub/esm](https://github.com/Biohub/esm)                             | Structure prediction |
| ESM3          | ✅ Supported | [Biohub/esm](https://github.com/Biohub/esm)                             | De novo design       |
| ProteinMPNN   | ✅ Supported | [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN)           | Sequence design      |
| LigandMPNN    | ✅ Supported | [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN)           | Sequence design      |
| SolubleMPNN   | ✅ Supported | [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN)           | Sequence design      |
| ThermoMPNN    | ✅ Supported | [Kuhlman-Lab/ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN)     | Property prediction  |
| ABodyBuilder3 | ⬜ Planned   | [Exscientia/abodybuilder3](https://github.com/Exscientia/abodybuilder3) | Structure prediction |
| BindCraft     | ✅ Supported | [martinpacesa/BindCraft](https://github.com/martinpacesa/BindCraft)     | De novo design       |
| Chai          | ✅ Supported | [chaidiscovery/chai-lab](https://github.com/chaidiscovery/chai-lab)     | Structure prediction |
| OpenDDE       | ⬜ Planned   | [aurekaresearch/OpenDDE](https://github.com/aurekaresearch/OpenDDE)     | Structure prediction |
| Protenix      | ⬜ Planned   | [bytedance/Protenix](https://github.com/bytedance/Protenix)             | Structure prediction |
| VESM          | ⬜ Planned   | [ntranoslab/vesm](https://github.com/ntranoslab/vesm)                   | Property prediction  |

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- A [Modal](https://modal.com) account

## Quickstart

Clone the repository and install dependencies:

```bash
git clone git@github.com:foldways/foldways-api.git
cd foldways-api
uv sync
```

Authenticate with Modal:

```bash
uv run modal setup
```

Create the Modal volume and stage model weights and other artifacts onto it:

```bash
uv run modal run setup_artifacts.py
```

Deploy the API:

```bash
uv run modal deploy app.py
```

## Usage

Modal displays the API URL in the [Modal dashboard](https://modal.com/apps),
under the app's `api` function. The URL has the format
`https://<workspace>--foldways-api.modal.run`. Interactive API docs are served
at `/docs`.

The examples below read the URL from `FOLDWAYS_API_URL`, so export it into your
shell first:

```bash
export FOLDWAYS_API_URL=https://<workspace>--foldways-api.modal.run
```

Nothing in the app reads this variable. It only keeps the commands below short,
and you can set it in `.env` instead if you prefer.

Submit a job. The response comes back with an id, while the workload runs on a
GPU in the background.

```bash
curl -X POST "$FOLDWAYS_API_URL/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "service": "boltz2",
    "name": "my-job",
    "params": {
      "sequences": [
        {"protein": {"id": "A", "sequence": "MELLKKLLEELKG"}}
      ]
    }
  }'
```

```json
{
  "id": "224f226c37b845d59a05b28405a4de0e",
  "name": "my-job",
  "service": "boltz2",
  "status": "pending",
  "created_at": "2025-07-22T16:00:00Z",
  "batch_id": null
}
```

Poll until the status is `complete`.

```bash
curl "$FOLDWAYS_API_URL/jobs/224f226c37b845d59a05b28405a4de0e"
```

Download the results as a zip, containing the predicted structures, confidence
scores, and the run log.

```bash
curl -OJ "$FOLDWAYS_API_URL/jobs/224f226c37b845d59a05b28405a4de0e/download"
```

To sweep one service over many parameter sets, submit a batch instead. It
returns a batch id plus a job id per member, and every job is validated before
any of them spawn.

```bash
curl -X POST "$FOLDWAYS_API_URL/batches" \
  -H "Content-Type: application/json" \
  -d '{
    "service": "boltz2",
    "name": "my-batch",
    "params": [
      {"sequences": [{"protein": {"id": "A", "sequence": "MELLKKLLEELKG"}}]},
      {"sequences": [{"protein": {"id": "A", "sequence": "MELLKKLLEELKA"}}]}
    ]
  }'
```

Then poll the batch for an aggregate status and the state of each job.

```bash
curl "$FOLDWAYS_API_URL/batches/<batch_id>"
```

Every parameter each service accepts is listed at `GET /services/{name}`, or
browsable at `/docs`.

## Usage with an agent

The API is self-describing, so an AI agent can drive it from your deployment URL
alone. It serves its full contract at `/openapi.json`, the deployed services and
their parameters at `/services`, and browsable docs at `/docs`.

To get an agent running jobs, give it your deployment URL and a prompt like:

> The Foldways API is at `<FOLDWAYS_API_URL>`. Read
> `<FOLDWAYS_API_URL>/openapi.json` for the contract and
> `<FOLDWAYS_API_URL>/services` for the available services. Submit a job with
> `POST /jobs`, poll `GET /jobs/{id}` until the status is `complete`, then
> `GET /jobs/{id}/download`. Add `"mock": true` to a request to test the flow
> without running on a GPU.

## Configuration

Configuration is optional. Foldways API ships with defaults for every service.

Each service takes four compute settings, prefixed with the service name: GPU
type, timeout, max containers, and scaledown window. To change any of them, copy
[`.env.example`](.env.example) to `.env` and override what you need. Defaults
are resolved in [`config.py`](config.py), and the valid values are listed in
[`constants.py`](constants.py).

These settings are read while `modal deploy` builds the app, not while it serves
traffic, so they are baked into the deployment. Change settings and redeploy for
them to take effect.

Deploying as shown above uses the defaults. To deploy with your overrides,
export `.env` into the shell first, so `modal deploy` can read them:

```bash
set -a; source .env; set +a
uv run modal deploy app.py
```

## Contributing

Issues and pull requests are welcome.

### Project layout

```
api/                FastAPI routers and request/response schemas
common/             Service registry, durable job/batch registries, shared helpers
services/           Per-tool Modal functions and images
mocks/              Fixture outputs served by mock job submissions
scripts/            Developer scripts, run as modules from the repo root
tests/              Pytest suite
docs/               Generated OpenAPI schema (openapi.json)
app.py              FastAPI app, exposed via modal.asgi_app
core.py             Modal app, images, and volume
setup_artifacts.py  Stages weights and mocks onto the volume
config.py           Environment-backed settings
constants.py        Static configuration
```

### Setup

Install the dev tooling:

```bash
make install-dev
```

### Branch names

Work happens on branches off `main`, named `<type>/<short-description>` with a
lowercase, hyphenated description:

| Prefix      | Use for                               | Example                  |
| ----------- | ------------------------------------- | ------------------------ |
| `feature/`  | New services or endpoints             | `feature/add-boltz2`     |
| `fix/`      | Bug fixes                             | `fix/batch-status-stuck` |
| `docs/`     | Documentation only                    | `docs/quickstart-volume` |
| `chore/`    | Tooling, dependencies, and CI         | `chore/bump-modal`       |
| `refactor/` | Restructuring with no behavior change | `refactor/job-registry`  |

### Checks

Linting and formatting use [Ruff](https://github.com/astral-sh/ruff).
Configuration lives in `pyproject.toml`, and CI runs the same lint, format,
typecheck, and test checks on every pull request.

Before opening a PR:

```bash
make format
make typecheck
make test
```

CI must pass before a pull request can be merged.

### OpenAPI schema

The API's OpenAPI schema is generated from the FastAPI app into
[`docs/openapi.json`](docs/openapi.json). It is committed so its diff shows
schema changes in review, and so the docs site has a fixed file to render.

When you change the API, regenerate it and commit the result with your change:

```bash
make openapi
```

### Mock fixtures

A job submitted with `"mock": true` returns a stored fixture instead of running
on a GPU, so the full submit, poll, and download lifecycle can be exercised
without any compute. Each service keeps its fixture under `mocks/<service>/`:

```
mocks/boltz2/request.json   The request that recorded the fixture
mocks/boltz2/output/        The recorded output, served as the mock result
```

Only `output/` is served, so `request.json` records provenance without appearing
in the download. Fixtures are staged onto the Modal volume by the artifact setup
step, alongside model weights.

To record or refresh a fixture, submit its request against a live deployment and
capture the result:

```bash
export FOLDWAYS_API_URL=https://<workspace>--foldways-api.modal.run
uv run python -m scripts.make_mocks boltz2
```

This runs real compute on a GPU. It reads `mocks/<service>/request.json`, waits
for the job, and overwrites `output/` with what it downloads. Review the diff,
commit it, then run `uv run modal run setup_artifacts.py` to stage the new
fixture on the volume.

### Releases

Release versions follow [Semantic Versioning](https://semver.org). The version
is only bumped when a release is cut, so the number describes compatibility
rather than counting commits. Bumping means editing two places,
`project.version` in [`pyproject.toml`](pyproject.toml) and `API_VERSION` in
[`constants.py`](constants.py).

To cut a release, bump the version in a `chore/release-<version>` branch and
merge it. Then publish the release on GitHub:

1. Go to the repository's **Releases** page and choose **Draft a new release**.
2. Under **Choose a tag**, type the new tag as `v<version>`, for example
   `v0.2.0`, and select **Create new tag on publish**. Leave the target as
   `main`, which tags the merge commit from step one.
3. Choose **Generate release notes** to build the notes from merged pull
   requests, then edit them to lead with anything that changes behavior.
4. Choose **Publish release**.

## License

[MIT](LICENSE)
