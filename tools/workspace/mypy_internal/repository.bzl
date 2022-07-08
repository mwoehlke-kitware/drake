# -*- python -*-

load("@drake//tools/workspace:github.bzl", "github_archive")

def mypy_internal_repository(
        name,
        mirrors = None):
    github_archive(
        name = name,
        repository = "python/mypy",
        commit = "v0.961",
        sha256 = "88f753229394032e8d266f04ea628b4cb3abc4d3cb6c182e52aecde66ab94bda",  # noqa
        build_file = "@drake//tools/workspace/mypy_internal:package.BUILD.bazel",  # noqa
        mirrors = mirrors,
    )
