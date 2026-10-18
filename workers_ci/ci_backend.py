"""Setuptools backend that writes wrangler.jsonc during installation."""

from __future__ import annotations

import os
from pathlib import Path

import setuptools.build_meta as orig
import write_wrangler


def _write() -> None:
    # backend-path puts this directory on sys.path, and an editable install
    # runs against the real checkout rather than a copied source tree.
    write_wrangler.write_wrangler_config(
        Path(__file__).resolve().parent.parent,
        os.environ.get("WRANGLER_CI_OVERRIDE_NAME"),
    )


def get_requires_for_build_wheel(config_settings=None):
    _write()
    return orig.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(config_settings=None):
    _write()
    return orig.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _write()
    return orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _write()
    return orig.build_wheel(
        wheel_directory, config_settings, metadata_directory
    )


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _write()
    return orig.prepare_metadata_for_build_editable(
        metadata_directory, config_settings
    )


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _write()
    return orig.build_editable(
        wheel_directory, config_settings, metadata_directory
    )
