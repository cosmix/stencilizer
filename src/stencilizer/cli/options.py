"""Typer option definitions for the bridge settings."""

import math
from typing import Annotated

import typer

from stencilizer.config import BridgeWidthScaling


def require_finite(value: float) -> float:
    """Reject NaN, which passes click's range check because every comparison with it is false."""
    if not math.isfinite(value):
        raise typer.BadParameter("must be a finite number")
    return value


BridgeWidthOption = Annotated[
    float,
    typer.Option(
        "--bridge-width",
        "-w",
        help="Bridge width as percent of a reference stroke of 10% of font UPM (30-110)",
        min=30.0,
        max=110.0,
        callback=require_finite,
    ),
]
WidthScalingOption = Annotated[
    BridgeWidthScaling,
    typer.Option(
        "--width-scaling",
        help="Variable fonts: keep bridge gaps fixed or scale them with each master's stroke weight",
    ),
]
ScalingStrengthOption = Annotated[
    float,
    typer.Option(
        "--scaling-strength",
        help="Proportional width scaling: 0 (fixed) to 100 (fully proportional)",
        min=0.0,
        max=100.0,
        callback=require_finite,
    ),
]
MinBridgeWidthOption = Annotated[
    float,
    typer.Option(
        "--min-bridge-width",
        help="Proportional width scaling: smallest gap as percent of a reference stroke (10-110)",
        min=10.0,
        max=110.0,
        callback=require_finite,
    ),
]
