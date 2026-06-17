# Security Policy

PulseShift is a research artifact: a static, client-side web page plus an offline Python
analysis pipeline. It runs no servers, stores no user data, has no authentication, and ships
no secrets or API keys. The browser app calls only two public, keyless APIs — the U.S.
National Weather Service and Open-Meteo — and collects nothing.

## Supported versions

Only the latest `main` is supported. Fixes land on `main`; there are no backports.

## Reporting a vulnerability

Please report privately rather than opening a public issue: use the repository's
**Security** tab → **Report a vulnerability** to open a private security advisory.

Expect an acknowledgement within a few days. Because there is no server or user data, the
realistic surface is limited to the dependency chain (audited in CI by `pip-audit`) and the
client-side handling of third-party API responses.
