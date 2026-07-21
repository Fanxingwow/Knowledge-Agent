# Security Policy

## Supported versions

Knowledge-Agent is currently pre-1.0. Security fixes are applied to the latest
commit on the `main` branch and to the most recent tagged release when
practical.

## Reporting a vulnerability

Do **not** create a public issue for a suspected vulnerability.

Use GitHub's private vulnerability reporting feature for this repository. If
that feature is unavailable, contact a maintainer privately and request a
secure reporting channel without including exploit details in public.

Please include:

- affected version or commit;
- deployment mode and relevant non-secret configuration;
- impact and affected trust boundary;
- minimal reproduction steps or proof of concept;
- suggested mitigation, if known.

Do not access data that is not yours, degrade shared services, or disclose
credentials while researching a report.

## Response process

Maintainers will acknowledge a complete report, validate impact, prepare a fix
and coordinate disclosure. Timelines depend on severity and reproducibility.
Credit is offered unless the reporter prefers to remain anonymous.

## Deployment responsibility

The example Compose files are development and single-host references. Public
deployments must add TLS, a reverse proxy, JWT validation, secret management,
network controls, backup/restore procedures, dependency patching and audit
monitoring.
