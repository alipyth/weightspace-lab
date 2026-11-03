# Security Policy

## Supported versions

Security fixes are applied to the latest version on the default branch.

## Reporting a vulnerability

Please do not publish credentials, access tokens, private model files, or exploitable details in a public issue. Contact the project maintainer through the contact links on [jahaniwww.com](https://jahaniwww.com) and include a clear description, reproduction steps, and the affected version.

## Deployment guidance

The application is designed to bind to `127.0.0.1` by default. It does not include user authentication. Do not expose it to a public network or reverse proxy without adding authentication, authorization, request-size limits, rate limits, and suitable isolation.

Hugging Face repositories are loaded with `trust_remote_code=False`. PyTorch checkpoint loading uses `weights_only=True`; do not add unsafe pickle loading for untrusted files. Treat model files and prompt content as untrusted input.
