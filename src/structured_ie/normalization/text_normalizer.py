"""Text normalization and OCR defect remediation for Polish legal documents."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedNIPCandidate:
    """Candidate NIP detected with its original text span and normalized 10-digit form."""

    raw_text: str
    normalized_nip: str
    start: int
    end: int
    is_valid_checksum: bool


class TextNormalizer:
    """Preprocesses and cleans noisy text extracted from OCR or scanned documents."""

    # Common OCR misspellings for NIP labels
    OCR_NIP_LABEL_PATTERN = re.compile(
        r"(?i)\b(N1P|N\s*!\s*P|N\s*l\s*P|N\s*I\s*P|NIP)\s*[:\-\s]",
    )

    # Hyphenated linebreaks in Polish words (e.g. 'ubez-\npieczający' -> 'ubezpieczający')
    SOFT_HYPHEN_BREAK_PATTERN = re.compile(r"(\w+)-\s*[\r\n]+\s*(\w+)", re.UNICODE)

    # General regex for Polish NIP with optional spaces, hyphens, and country code
    # Matches patterns like 'NIP: 525 12 34 567', 'PL 525-123-45-67', '5251234567'
    NIP_EXTRACT_PATTERN = re.compile(
        r"(?i)(?:(?:N1P|N\s*!\s*P|N\s*l\s*P|N\s*I\s*P|NIP)\s*[:\-\s]?\s*)?"
        r"(?:PL\s*)?(\d(?:[\s\-\.]?\d){8,11})\b"
    )

    @classmethod
    def clean_ocr_linebreaks(cls, text: str) -> str:
        """Join words split across line breaks by hyphens."""
        return cls.SOFT_HYPHEN_BREAK_PATTERN.sub(r"\1\2", text)

    @classmethod
    def normalize_nip_string(cls, candidate: str) -> str:
        """Strip non-digits and optional country code 'PL' from NIP candidate."""
        candidate_clean = re.sub(r"(?i)^pl", "", candidate.strip())
        digits = re.sub(r"\D", "", candidate_clean)
        return digits

    @classmethod
    def find_nip_candidates(cls, text: str) -> list[NormalizedNIPCandidate]:
        """Scan text and identify all candidate 10-digit NIPs, computing offsets and checksums.

        Args:
            text: Raw input document string.

        Returns:
            List of NormalizedNIPCandidate instances.
        """
        from structured_ie.normalization.checksums import validate_nip_checksum

        candidates: list[NormalizedNIPCandidate] = []
        for match in cls.NIP_EXTRACT_PATTERN.finditer(text):
            group_val = match.group(1)
            raw_match_text = match.group(0)
            digits = cls.normalize_nip_string(group_val)

            if len(digits) == 10:
                # Find start and end of the digit group within text
                group_start = match.start(1)
                group_end = match.end(1)
                is_valid = validate_nip_checksum(digits)
                candidates.append(
                    NormalizedNIPCandidate(
                        raw_text=raw_match_text.strip(),
                        normalized_nip=digits,
                        start=group_start,
                        end=group_end,
                        is_valid_checksum=is_valid,
                    )
                )

        return candidates
