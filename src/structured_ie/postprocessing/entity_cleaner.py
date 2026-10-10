"""Post-processing and normalization for extracted named entities."""

import logging
import re

from structured_ie.schemas.models import Entity

logger = logging.getLogger(__name__)


class EntityPostprocessor:
    """Cleans, normalizes, and disambiguates entity spans produced by zero-shot models.

    Addresses systematic failure modes:
    1. Removes label trigger prefixes (e.g. 'NIP: 5251234567' -> '5251234567').
    2. Discards pure trigger words (e.g. entity 'NIP' or 'REGON' with no digits).
    3. Disambiguates Polish tax numbers by digit length (10 digits = NIP, 9/14 digits = REGON).
    4. Normalizes company surface forms by trimming generic introductory nouns.
    5. Preserves character offset synchronization (text[start:end] == entity.text).
    """

    def __init__(
        self,
        clean_identifiers: bool = True,
        disambiguate_lengths: bool = True,
        clean_company_descriptors: bool = True,
    ) -> None:
        """Initialize post-processor.

        Args:
            clean_identifiers: Whether to extract exact digit sequences for NIP/REGON.
            disambiguate_lengths: Whether to enforce 10-digit NIP and 9/14-digit REGON rules.
            clean_company_descriptors: Whether to trim generic corporate prefixes (Spółka, etc.).
        """
        self.clean_identifiers = clean_identifiers
        self.disambiguate_lengths = disambiguate_lengths
        self.clean_company_descriptors = clean_company_descriptors

    def clean_entities(
        self,
        entities: list[Entity],
        text: str,
    ) -> list[Entity]:
        """Convenience alias for postprocess."""
        return self.postprocess(entities, text)

    def postprocess(self, entities: list[Entity], text: str) -> list[Entity]:
        """Process list of raw entities against original source text.

        Args:
            entities: Extracted Entity objects from NER model.
            text: Original raw input string.

        Returns:
            List of normalized and validated Entity objects.
        """
        cleaned: list[Entity] = []

        for ent in entities:
            lbl = ent.label.strip()

            if self.clean_identifiers and lbl in ("NIP", "REGON"):
                processed_ent = self._clean_identifier(ent, text)
                if processed_ent is not None:
                    cleaned.append(processed_ent)
            elif self.clean_company_descriptors and lbl == "company":
                processed_ent = self._clean_company(ent, text)
                cleaned.append(processed_ent)
            else:
                cleaned.append(ent)

        return cleaned

    def _clean_identifier(self, ent: Entity, text: str) -> Entity | None:
        """Isolate digit span, disambiguate type, and adjust exact character offsets."""
        digits = re.findall(r"\d+", ent.text)
        if not digits:
            logger.debug(
                "Discarding trigger-word entity '%s' for label '%s' (no digits found).",
                ent.text,
                ent.label,
            )
            return None

        digit_str = "".join(digits)

        # Discard fragments with fewer than 9 digits (e.g. OCR artefacts like '1' from 'N1P')
        if len(digit_str) < 9:
            logger.debug(
                "Discarding short digit sequence '%s' for label '%s' (< 9 digits).",
                digit_str,
                ent.label,
            )
            return None

        # Check if preceding context indicates KRS rather than NIP/REGON
        before_ctx = text[max(0, ent.start - 30) : ent.start].lower()
        if "krs" in before_ctx and "nip" not in before_ctx:
            logger.debug(
                "Discarding 10-digit number '%s' because preceding context indicates KRS.",
                digit_str,
            )
            return None

        # Disambiguate official Polish legal identifier lengths
        target_label = ent.label
        if self.disambiguate_lengths:
            if len(digit_str) == 10:
                target_label = "NIP"
            elif len(digit_str) in (9, 14):
                target_label = "REGON"

        # Locate exact position in original text within or near the entity span
        # If the extracted span contains formatted digits (e.g. '525-123-45-67' or 'PL 525 12 34 567'),
        # determine start from the first digit and end from the last digit.
        first_digit_m = re.search(r"\d", ent.text)
        last_digit_matches = list(re.finditer(r"\d", ent.text))
        if first_digit_m and last_digit_matches:
            rel_start = first_digit_m.start()
            rel_end = last_digit_matches[-1].end()
            start = ent.start + rel_start
            end = ent.start + rel_end
        else:
            idx_in_span = ent.text.find(digit_str)
            if idx_in_span != -1:
                start = ent.start + idx_in_span
                end = start + len(digit_str)
            else:
                start = text.find(digit_str, max(0, ent.start - 10))
                if start == -1:
                    start = text.find(digit_str)
                end = start + len(digit_str) if start != -1 else ent.end

        return Entity(
            text=digit_str,
            label=target_label,
            score=ent.score,
            start=start,
            end=end,
        )

    def _clean_company(self, ent: Entity, text: str) -> Entity:
        """Optionally strip boilerplate corporate introductory words and role prefixes."""
        pattern = r"^(Spółka|Firma|Przedsiębiorstwo|Startup|Ubezpieczający|Ubezpieczonego|Ubezpieczony|Klient|Mocodawca|Zleceniodawca|Poszkodowany)\s+"
        match = re.match(pattern, ent.text, flags=re.IGNORECASE)
        if not match:
            return ent

        prefix_len = match.end()
        candidate_text = ent.text[prefix_len:].strip()

        # If stripping would leave an empty or too short string, retain original
        if len(candidate_text) < 3:
            return ent

        # Update offsets to reflect trimmed substring
        new_start = ent.start + (len(ent.text) - len(ent.text.lstrip())) + prefix_len
        new_end = new_start + len(candidate_text)

        if text[new_start:new_end] == candidate_text:
            return Entity(
                text=candidate_text,
                label=ent.label,
                score=ent.score,
                start=new_start,
                end=new_end,
            )

        return ent
