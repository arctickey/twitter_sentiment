#!/bin/sh
git config core.editor "code --wait"
git config --global user.name "${GIT_USER}"
git config --global user.email "${GIT_EMAIL}"
poetry self update
poetry install --no-interaction