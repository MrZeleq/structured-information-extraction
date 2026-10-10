"""Unit tests for TextNormalizer and OCR defect remediation."""

from structured_ie.normalization.text_normalizer import TextNormalizer


def test_clean_ocr_linebreaks() -> None:
    raw_text = "Mocodawca ubez-\npieczający upoważnia bro-\n  kera do działania."
    cleaned = TextNormalizer.clean_ocr_linebreaks(raw_text)
    assert "ubezpieczający" in cleaned
    assert "brokera" in cleaned


def test_normalize_nip_string() -> None:
    assert TextNormalizer.normalize_nip_string("525 000 77 38") == "5250007738"
    assert TextNormalizer.normalize_nip_string("PL 525-000-77-38") == "5250007738"
    assert TextNormalizer.normalize_nip_string("525-00-07-738") == "5250007738"


def test_find_nip_candidates() -> None:
    text = (
        "Dokument wystawiony dla: Ubezpieczający (NIP: 525 000 77 38) oraz "
        "Broker: N1P 526-025-10-49."
    )
    candidates = TextNormalizer.find_nip_candidates(text)
    assert len(candidates) == 2

    # First candidate: 5250007738 (PKO BP, valid checksum)
    assert candidates[0].normalized_nip == "5250007738"
    assert candidates[0].is_valid_checksum is True

    # Second candidate: 5260251049 (PZU, valid checksum, OCR typo N1P)
    assert candidates[1].normalized_nip == "5260251049"
    assert candidates[1].is_valid_checksum is True
