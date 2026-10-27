from __future__ import annotations

import math


def format_significant(value: float, digits: int = 3, *, scientific: bool = False) -> str:
    """Format a finite scalar with exactly ``digits`` significant digits."""
    value = float(value)
    if not math.isfinite(value):
        return "nan"
    if value == 0.0:
        return f"{value:.{digits - 1}e}" if scientific else "0." + "0" * (digits - 1)
    if scientific:
        return f"{value:.{digits - 1}e}"
    rounded = float(f"{value:.{digits}g}")
    exponent = math.floor(math.log10(abs(rounded)))
    decimals = max(0, digits - 1 - exponent)
    return f"{rounded:.{decimals}f}"


def format_correlation(value: float) -> str:
    """Three-significant-digit formatter for bounded correlation values."""
    return format_significant(value, 3)


def format_p_value(value: float) -> str:
    """Three significant digits, using scientific notation for small p-values."""
    value = float(value)
    return format_significant(value, 3, scientific=value == 0.0 or abs(value) < 1e-3)
