---
title: Halohalo Dashboard
emoji: 🍧
colorFrom: green
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: true
license: mit
short_description: Dashboard of Philippine Datasets within Sapin-sapin Project
sdk_version: 5.43.1
---

# halohalo dashboard

A live dashboard of everything in the
[sapinsapin](https://huggingface.co/sapinsapin) org: Philippine-language
corpora (speech, web text, literary text) and models finetuned on them.

The app holds no hardcoded repo list — each page load queries the Hub API, so
new datasets and models appear automatically. Private repos are included in
the counts and listed as rows with their names withheld.

Configuration:

- `DASHBOARD_ORG` (env, optional) — org to display; defaults to `sapinsapin`.
- `HF_TOKEN` (Space secret, optional) — a **read-scoped** token that can see
  the org's private repos, making the private rows fully live. Without it the
  app falls back to `private_manifest.json`, a name-free snapshot holding only
  the count of private repos per type (regenerate it when private repos are
  added or removed — or just set the secret).
