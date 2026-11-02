#!/usr/bin/env bash

RUN_ID=$(gh run list --workflow=centinela-train.yml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run download "$RUN_ID" -n "centinela-run-$RUN_ID" -D ./local_run

ls -la ./local_run
cat ./local_run/metrics.json | jq .
