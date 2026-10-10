"""Unit tests for Polish identifier checksum verification (NIP, REGON, KRS)."""

import pytest

from structured_ie.normalization.checksums import (
    validate_krs,
    validate_nip_checksum,
    validate_regon_checksum,
)


@pytest.mark.parametrize(
    ("nip", "expected"),
    [
        ("5250007738", True),  # PKO BP
        ("5260251049", True),  # PZU SA
        ("7740001454", True),  # PKN Orlen
        ("5270103391", True),  # Miasto st. Warszawa
        ("525-000-77-38", True),  # Formatted with hyphens
        ("PL 525 000 77 38", True),  # Formatted with spaces and PL
        ("1234567890", False),  # Invalid check digit
        ("5250007739", False),  # Corrupted last digit
        ("12345", False),  # Too short
        ("1234567890123", False),  # Too long
        ("", False),
        (None, False),
        ("abcdefghij", False),
    ],
)
def test_validate_nip_checksum(nip: str | None, expected: bool) -> None:
    assert validate_nip_checksum(nip) == expected


@pytest.mark.parametrize(
    ("regon", "expected"),
    [
        ("012345678", False),  # Invalid 9-digit
        ("12345", False),  # Too short
        (None, False),
        ("", False),
    ],
)
def test_validate_regon_checksum(regon: str | None, expected: bool) -> None:
    assert validate_regon_checksum(regon) == expected


@pytest.mark.parametrize(
    ("krs", "expected"),
    [
        ("0000012345", True),
        ("12345", True),
        ("0000123456789", False),  # Too long (>10 digits)
        (None, False),
        ("", False),
    ],
)
def test_validate_krs(krs: str | None, expected: bool) -> None:
    assert validate_krs(krs) == expected
