#!/bin/sh

export MAVEN_USER=$FED_MAVEN_REPO_USERNAME
export MAVEN_PASSWORD=$FED_MAVEN_REPO_PASSWORD
REPO_URL="https://packages.jetbrains.team/maven/p/fed/maven"
ENGINE_TARGET="//engine/java/src/main/org/jetbrains/tff/engine:publish"
bazel run --stamp --define "maven_repo=$REPO_URL" --define gpg_sign=false $ENGINE_TARGET --java_language_version=17 --java_runtime_version=17
