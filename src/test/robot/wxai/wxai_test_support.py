"""The WidowX AI expruns as checked in, resolved without Config (defaults, then the run)."""

import pathlib

import yaml

EXPRUNS = pathlib.Path(__file__).resolve().parents[4] / "data" / "expruns"


def load_exp(family, run):
    with open(EXPRUNS / family / f"_defaults_{family}.yaml") as handle:
        values = yaml.safe_load(handle) or {}
    with open(EXPRUNS / family / f"{run}.yaml") as handle:
        values |= yaml.safe_load(handle) or {}
    return values


def robot_exp(run="position_controller_fake_wxai_00"):
    return load_exp("robot_wxai", run)


def leader_exp():
    return load_exp("controllers", "wxai_leader_fake_00")
