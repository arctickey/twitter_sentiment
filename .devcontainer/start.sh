#!/bin/sh
git config core.editor "code --wait"
git config --global user.name "${GIT_USER}"
git config --global user.email "${GIT_EMAIL}"
git config --global --add url."git@github.com:".insteadOf "https://github.com/"
poetry self update
poetry install --no-interaction
