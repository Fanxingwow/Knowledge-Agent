# Roadmap

The roadmap communicates direction, not a delivery commitment.

## Reliability

- Make embedding and parser readiness probes non-blocking under active load.
- Add explicit import cancellation and dead-letter administration.
- Publish backup, restore and disaster-recovery exercises.

## RAG quality

- Expand multilingual evaluation fixtures and citation-grounding metrics.
- Add pluggable chunking, parser, embedding and reranker interfaces.
- Add query tracing and retrieval-debug views without exposing private text.

## Platform

- Provide reference Kubernetes and Helm deployments.
- Add OpenTelemetry traces and dashboards.
- Support organization, role and corpus administration through the web UI.
- Publish signed container images and reproducible release artifacts.

## Developer experience

- Add a guided first-run configuration command.
- Reduce CPU-only image size and inference startup time.
- Publish stable extension contracts before 1.0.
