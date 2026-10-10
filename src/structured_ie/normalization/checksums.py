"""Mathematical checksum verification for Polish corporate and tax identifiers (NIP, REGON, KRS)."""

import re


def validate_nip_checksum(nip: str | None) -> bool:
    """Validate Polish Tax Identification Number (NIP) using Modulo 11 checksum algorithm.

    NIP consists of 10 digits:
    - Digits 1-9: registration digits
    - Digit 10: check digit
    Weights applied: [6, 5, 7, 2, 3, 4, 5, 6, 7]
    The control digit cannot be 10.

    Args:
        nip: Candidate NIP string (may contain whitespace or hyphens).

    Returns:
        True if valid 10-digit NIP with correct Modulo 11 check digit, False otherwise.
    """
    if not nip:
        return False

    digits_only = re.sub(r"\D", "", nip)
    if len(digits_only) != 10:
        return False

    weights = (6, 5, 7, 2, 3, 4, 5, 6, 7)
    digits = [int(d) for d in digits_only]

    checksum = sum(d * w for d, w in zip(digits[:9], weights, strict=True)) % 11
    if checksum == 10:
        return False

    return checksum == digits[9]


def validate_regon_checksum(regon: str | None) -> bool:
    """Validate Polish National Business Registry Number (REGON 9 or 14 digits).

    Args:
        regon: Candidate REGON string.

    Returns:
        True if valid REGON with correct checksum, False otherwise.
    """
    if not regon:
        return False

    digits_only = re.sub(r"\D", "", regon)
    digits = [int(d) for d in digits_only]

    if len(digits) == 9:
        weights = (8, 9, 2, 3, 4, 5, 6, 7)
        checksum = sum(d * w for d, w in zip(digits[:8], weights, strict=True)) % 11
        if checksum == 10:
            checksum = 0
        return checksum == digits[8]

    if len(digits) == 14:
        # First 9 digits must also be valid
        if not validate_regon_checksum(digits_only[:9]):
            return False
        weights14 = (2, 4, 8, 5, 0, 9, 7, 3, 6, 1, 2, 4, 8)
        checksum = sum(d * w for d, w in zip(digits[:13], weights14, strict=True)) % 11
        if checksum == 10:
            checksum = 0
        return checksum == digits[13]

    return False


def validate_krs(krs: str | None) -> bool:
    """Validate National Court Register (KRS) number format (1 to 10 digits).

    Args:
        krs: Candidate KRS string.

    Returns:
        True if valid KRS digits format, False otherwise.
    """
    if not krs:
        return False
    digits_only = re.sub(r"\D", "", krs)
    return 1 <= len(digits_only) <= 10
