"""Neuro-symbolic resolver for extracting brokers, clients, and binding tax identifiers (NIP)."""

import logging
from typing import Any

from pydantic import BaseModel, Field

from structured_ie.normalization.checksums import validate_nip_checksum
from structured_ie.schemas.models import Entity, Relation

logger = logging.getLogger(__name__)


class BrokerParty(BaseModel):
    """Structured information for the insurance broker."""

    name: str = Field(description="Legal surface name of the broker company")
    nip: str | None = Field(default=None, description="10-digit tax identification number")
    name_span: tuple[int, int] = Field(description="Character offsets (start, end) of broker name")
    nip_span: tuple[int, int] | None = Field(
        default=None, description="Character offsets (start, end) of broker NIP"
    )
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")
    is_nip_valid: bool | None = Field(
        default=None, description="Whether NIP passed Modulo 11 checksum verification"
    )
    raw_nip_text: str | None = Field(
        default=None, description="Raw surface form of NIP as detected in text"
    )


class ClientParty(BaseModel):
    """Structured information for the insured / client."""

    name: str = Field(description="Legal surface name of the client company")
    nip: str | None = Field(default=None, description="10-digit tax identification number")
    name_span: tuple[int, int] = Field(description="Character offsets (start, end) of client name")
    nip_span: tuple[int, int] | None = Field(
        default=None, description="Character offsets (start, end) of client NIP"
    )
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")
    is_nip_valid: bool | None = Field(
        default=None, description="Whether NIP passed Modulo 11 checksum verification"
    )
    raw_nip_text: str | None = Field(
        default=None, description="Raw surface form of NIP as detected in text"
    )


class BrokerClientExtraction(BaseModel):
    """Consolidated business extraction result identifying broker and client parties."""

    broker: BrokerParty | None = Field(default=None, description="Identified insurance broker")
    client: ClientParty | None = Field(default=None, description="Identified client/insured party")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Overall pipeline confidence"
    )
    needs_review: bool = Field(
        default=False, description="Flag indicating human-in-the-loop review is recommended"
    )
    review_reasons: list[str] = Field(
        default_factory=list, description="Audit log of reasons why review is required"
    )

    @property
    def confidence_score(self) -> float:
        """Alias for confidence to match standard naming conventions."""
        return self.confidence

    def to_dict(self) -> dict[str, Any]:
        """Serialize extraction to standard dictionary format."""
        return {
            "broker": self.broker.model_dump() if self.broker else None,
            "client": self.client.model_dump() if self.client else None,
            "confidence": round(self.confidence, 4),
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
        }


class BrokerClientResolver:
    """Disambiguates corporate roles (broker vs client) and binds NIP identifiers syntactically.

    Implements a two-phase neuro-symbolic resolution:
    1. Syntactic NIP-to-Company Binding: Matches 10-digit NIPs to their owning company spans
       using parenthetical co-location and legal entity registration clause proximity.
    2. Competitive Role Scoring: Applies lexical and syntactic indicators under a mutual
       exclusivity constraint to rank which entity serves as the broker and which as the client.
    """

    # Generic morphological stems identifying insurance brokerage activities
    DEFAULT_GENERIC_BROKER_ROOTS: tuple[str, ...] = (
        "broker",
        "brokersk",
        "kancelaria brokerska",
        "kancelarii brokerskiej",
        "dom brokerski",
        "domu brokerskiego",
        "makler",
        "maklersk",
        "pośrednik ubezpieczeniowy",
        "pośrednictwo ubezpieczeniowe",
    )

    # Backward compatibility alias
    DEFAULT_BROKER_NAME_ROOTS: tuple[str, ...] = DEFAULT_GENERIC_BROKER_ROOTS

    DEFAULT_BROKER_CONTEXT_CUES: tuple[str, ...] = (
        "jako broker",
        "za pośrednictwem brokera",
        "brokerowi",
        "brokerem",
        "udziela pełnomocnictwa brokerowi",
        "udzielam pełnomocnictwa brokerowi",
        "wyznacza brokera",
        "obsługę brokerską",
        "obsłudze brokerskiej",
        "pośrednik ubezpieczeniowy",
        "pełnomocnik",
        "broker ubezpieczeniowy",
    )

    DEFAULT_CLIENT_CONTEXT_CUES: tuple[str, ...] = (
        "ubezpieczający",
        "ubezpieczającego",
        "ubezpieczony",
        "mocodawca",
        "mocodawcy",
        "zleceniodawca",
        "zleceniodawcy",
        "klient",
        "klienta",
        "w imieniu mocodawcy",
        "na rzecz klienta",
        "poszkodowany",
        "poszkodowanego",
    )

    def __init__(
        self,
        broker_name_roots: tuple[str, ...] | None = None,
        broker_registry: set[str] | list[str] | tuple[str, ...] | None = None,
        broker_context_cues: tuple[str, ...] | None = None,
        client_context_cues: tuple[str, ...] | None = None,
        max_registration_distance: int = 350,
    ) -> None:
        """Initialize resolver with lexical and contextual rules.

        Args:
            broker_name_roots: Generic morphological roots identifying brokers.
            broker_registry: Optional external set of known broker names/stems (e.g. from KNF).
            broker_context_cues: Lexical triggers indicating the presence of a broker.
            client_context_cues: Lexical triggers indicating the presence of a client.
            max_registration_distance: Maximum character distance for legal entity registration clauses.
        """
        self.broker_name_roots = broker_name_roots or self.DEFAULT_GENERIC_BROKER_ROOTS
        self.broker_registry: set[str] = (
            {r.lower().strip() for r in broker_registry} if broker_registry else set()
        )
        self.broker_context_cues = broker_context_cues or self.DEFAULT_BROKER_CONTEXT_CUES
        self.client_context_cues = client_context_cues or self.DEFAULT_CLIENT_CONTEXT_CUES
        self.max_registration_distance = max_registration_distance

    def bind_nips_to_companies(
        self,
        companies: list[Entity],
        nips: list[Entity],
        text: str,
    ) -> dict[str, Entity]:
        """Bind each company to its exact corresponding NIP entity.

        Applies a three-step matching priority:
        1. Parenthetical attachment: `Company Name (NIP: 1234567890)`
        2. Legal entity registration block / clause: addresses, KRS, REGON, court info
           up to max_registration_distance (e.g. 350 chars), ensuring no other company
           intervenes in between.
        3. Nearest unassigned candidate fallback.

        Args:
            companies: Extracted company entities.
            nips: Extracted 10-digit NIP entities.
            text: Full raw text of the document.

        Returns:
            Dictionary mapping `company.text` to the bound `Entity` (NIP).
        """
        if not companies or not nips:
            return {}

        assigned_nips: set[int] = set()
        company_bindings: dict[str, Entity] = {}

        # Pass 1: Parenthetical attachment e.g. "Firma X (NIP: 1234567890)"
        for comp in companies:
            for nip in nips:
                if id(nip) in assigned_nips:
                    continue

                if comp.end <= nip.start:
                    between = text[comp.end : nip.start]
                    # Up to 40 chars containing opening parenthesis or NIP label
                    if (
                        len(between) <= 40
                        and ("(" in between or "nip" in between.lower() or ":" in between)
                        and ")" not in between
                    ):
                        # Ensure no other company is between
                        if not any(
                            other != comp and other.start >= comp.end and other.end <= nip.start
                            for other in companies
                        ):
                            company_bindings[comp.text] = nip
                            assigned_nips.add(id(nip))
                            break

        # Pass 2: Legal Entity Registration Block (Addresses, KRS, REGON, colons)
        for comp in companies:
            if comp.text in company_bindings:
                continue

            for nip in nips:
                if id(nip) in assigned_nips:
                    continue

                # Case A: Company precedes NIP (standard legal entity description)
                if comp.end <= nip.start:
                    dist = nip.start - comp.end
                    if dist <= self.max_registration_distance:
                        # Ensure NO other company appears between comp.end and nip.start
                        has_intervening_company = any(
                            other != comp and other.start >= comp.end and other.end <= nip.start
                            for other in companies
                        )
                        if not has_intervening_company:
                            between = text[comp.end : nip.start]
                            between_lower = between.lower()
                            is_registration_clause = (
                                any(
                                    marker in between_lower
                                    for marker in (
                                        "nip",
                                        "krs",
                                        "regon",
                                        "siedzib",
                                        "adres",
                                        "ul.",
                                        "kapitał",
                                        "sąd",
                                        "rejestr",
                                        ":",
                                        ",",
                                        ";",
                                        "\n",
                                    )
                                )
                                and between.count("\n\n") == 0
                            )
                            if is_registration_clause:
                                company_bindings[comp.text] = nip
                                assigned_nips.add(id(nip))
                                break

                # Case B: NIP precedes Company (e.g. structured key-value: 'NIP: 525... Firma ABC')
                elif nip.end <= comp.start:
                    dist = comp.start - nip.end
                    if dist <= 60:
                        has_intervening_company = any(
                            other != comp and other.start >= nip.end and other.end <= comp.start
                            for other in companies
                        )
                        if not has_intervening_company:
                            company_bindings[comp.text] = nip
                            assigned_nips.add(id(nip))
                            break

        # Pass 3: Minimal character distance among remaining candidates
        remaining_companies = [c for c in companies if c.text not in company_bindings]
        remaining_nips = [n for n in nips if id(n) not in assigned_nips]

        for comp in remaining_companies:
            if not remaining_nips:
                break

            # Find closest unassigned NIP with no intervening company
            valid_candidates = []
            for n in remaining_nips:
                c_start, c_end = (comp.end, n.start) if comp.end <= n.start else (n.end, comp.start)
                has_intervening = any(
                    other != comp and other.start >= c_start and other.end <= c_end
                    for other in companies
                )
                if not has_intervening:
                    dist = (
                        abs(n.start - comp.end) if comp.end <= n.start else abs(comp.start - n.end)
                    )
                    valid_candidates.append((dist, n))

            if valid_candidates:
                valid_candidates.sort(key=lambda x: x[0])
                closest_nip = valid_candidates[0][1]
                company_bindings[comp.text] = closest_nip
                remaining_nips.remove(closest_nip)
                assigned_nips.add(id(closest_nip))

        return company_bindings

    def _score_company(self, comp: Entity, text: str) -> tuple[float, float]:
        """Compute relative (broker_score, client_score) for a candidate company."""
        name_lower = comp.text.lower()
        ctx_start = max(0, comp.start - 120)
        ctx_end = min(len(text), comp.end + 120)
        ctx_lower = text[ctx_start:ctx_end].lower()

        b_score = 0.0
        c_score = 0.0

        # 1. External registry check (if configured)
        if any(reg in name_lower for reg in self.broker_registry):
            b_score += 5.0

        # 2. Generic morphological/name-level cues
        for root in self.broker_name_roots:
            if root in name_lower:
                b_score += 4.0

        # 3. Local context window cues
        if any(cue in ctx_lower for cue in self.broker_context_cues):
            b_score += 2.0
        if any(cue in ctx_lower for cue in self.client_context_cues):
            c_score += 2.0

        # 4. Direct preceding markers
        before_text = text[max(0, comp.start - 60) : comp.start].lower()
        if any(
            w in before_text
            for w in [
                "kancelaria brokerska",
                "kancelarii brokerskiej",
                "dom brokerski",
                "domu brokerskiego",
                "broker:",
                "brokera:",
                "brokerowi",
                "wyznaczony broker",
                "wyznaczonemu brokerowi",
                "jako broker",
                "za pośrednictwem brokera",
                "obsługę brokerską spółce",
                "obsłudze brokerskiej spółce",
                "upoważnia",
                "upoważniam",
                "udziela pełnomocnictwa",
                "wyznacza",
                "jedynym wyłącznym brokerem",
                "zapytanie składa broker",
            ]
        ):
            b_score += 5.0
        elif "broker" in before_text:
            b_score += 3.0

        if any(
            w in before_text
            for w in [
                "na rzecz klienta",
                "klient:",
                "klienta",
                "ubezpieczający:",
                "ubezpieczającego",
                "ubezpieczonego",
                "w imieniu mocodawcy",
                "mocodawca",
                "zleceniodawca",
                "poszkodowanego",
                "zlecenia od firmy",
                "odwołuje pełnomocnictwo",
            ]
        ):
            c_score += 5.0

        # 5. Direct following clause markers
        after_text = text[comp.end : min(len(text), comp.end + 70)].lower()
        if any(
            w in after_text
            for w in [
                "działając jako broker",
                "działający jako broker",
                "sprawuje broker",
                "jako broker",
                "reprezentując ubezpieczającego",
                "reprezentujący ubezpieczającego",
                "w imieniu mocodawcy",
            ]
        ):
            b_score += 4.0

        if any(
            w in after_text
            for w in [
                "upoważnia",
                "udziela pełnomocnictwa",
                "powierza obsługę",
                "odwołuje pełnomocnictwo",
                "reprezentowany przez",
            ]
        ):
            c_score += 4.0

        return b_score, c_score

    def resolve(
        self,
        entities: list[Entity],
        text: str,
    ) -> tuple[list[Entity], list[Relation]]:
        """Resolve corporate roles and construct standard Entity and Relation objects.

        Args:
            entities: Extracted and cleaned Named Entities.
            text: Raw input text.

        Returns:
            Tuple of (re-typed entities with 'broker' and 'client' labels, extracted relations).
        """
        companies = [e for e in entities if e.label in ("company", "broker", "client")]
        nips = [e for e in entities if e.label == "NIP"]

        if not companies:
            return entities, []

        comp_to_nip = self.bind_nips_to_companies(companies, nips, text)

        broker_comp: Entity | None = None
        client_comp: Entity | None = None

        if len(companies) >= 2:
            scored = []
            for c in companies[:2]:
                b_s, c_s = self._score_company(c, text)
                delta = b_s - c_s
                scored.append((delta, c))

            # Sort descending by delta (highest broker score first)
            scored.sort(key=lambda x: x[0], reverse=True)
            broker_comp = scored[0][1]
            client_comp = scored[1][1]
        elif len(companies) == 1:
            b_s, c_s = self._score_company(companies[0], text)
            if b_s >= c_s:
                broker_comp = companies[0]
            else:
                client_comp = companies[0]

        # Re-type entities with specific business roles
        out_entities: list[Entity] = []
        for e in entities:
            if broker_comp and e.text == broker_comp.text and e.start == broker_comp.start:
                out_entities.append(
                    Entity(text=e.text, label="broker", score=e.score, start=e.start, end=e.end)
                )
            elif client_comp and e.text == client_comp.text and e.start == client_comp.start:
                out_entities.append(
                    Entity(text=e.text, label="client", score=e.score, start=e.start, end=e.end)
                )
            else:
                out_entities.append(e)

        # Generate canonical relation triples
        out_relations: list[Relation] = []
        if broker_comp and broker_comp.text in comp_to_nip:
            bnip = comp_to_nip[broker_comp.text]
            out_relations.append(
                Relation(
                    source=broker_comp.text,
                    relation="has NIP",
                    target=bnip.text,
                    score=1.0,
                    source_span=(broker_comp.start, broker_comp.end),
                    target_span=(bnip.start, bnip.end),
                )
            )

        if client_comp and client_comp.text in comp_to_nip:
            cnip = comp_to_nip[client_comp.text]
            out_relations.append(
                Relation(
                    source=client_comp.text,
                    relation="has NIP",
                    target=cnip.text,
                    score=1.0,
                    source_span=(client_comp.start, client_comp.end),
                    target_span=(cnip.start, cnip.end),
                )
            )

        if broker_comp and client_comp:
            out_relations.append(
                Relation(
                    source=broker_comp.text,
                    relation="broker for",
                    target=client_comp.text,
                    score=1.0,
                    source_span=(broker_comp.start, broker_comp.end),
                    target_span=(client_comp.start, client_comp.end),
                )
            )

        return out_entities, out_relations

    def extract_structured(
        self,
        entities: list[Entity],
        text: str,
    ) -> BrokerClientExtraction:
        """Produce structured high-level BrokerClientExtraction model with audit and validation metadata.

        Args:
            entities: Extracted entities.
            text: Original document text.

        Returns:
            BrokerClientExtraction with broker and client party details, validation flags,
            and review recommendations.
        """
        retyped_ents, relations = self.resolve(entities=entities, text=text)
        broker_ent = next((e for e in retyped_ents if e.label == "broker"), None)
        client_ent = next((e for e in retyped_ents if e.label == "client"), None)

        broker_nip = next(
            (
                r.target
                for r in relations
                if broker_ent and r.source == broker_ent.text and r.relation == "has NIP"
            ),
            None,
        )
        broker_nip_ent = next(
            (e for e in retyped_ents if e.label == "NIP" and e.text == broker_nip), None
        )

        client_nip = next(
            (
                r.target
                for r in relations
                if client_ent and r.source == client_ent.text and r.relation == "has NIP"
            ),
            None,
        )
        client_nip_ent = next(
            (e for e in retyped_ents if e.label == "NIP" and e.text == client_nip), None
        )

        # Validate NIP checksums via Modulo 11
        is_broker_nip_valid = validate_nip_checksum(broker_nip) if broker_nip else None
        is_client_nip_valid = validate_nip_checksum(client_nip) if client_nip else None

        broker_party = None
        if broker_ent:
            broker_party = BrokerParty(
                name=broker_ent.text,
                nip=broker_nip,
                name_span=(broker_ent.start, broker_ent.end),
                nip_span=(broker_nip_ent.start, broker_nip_ent.end) if broker_nip_ent else None,
                score=broker_ent.score,
                is_nip_valid=is_broker_nip_valid,
                raw_nip_text=text[broker_nip_ent.start : broker_nip_ent.end]
                if broker_nip_ent
                else None,
            )

        client_party = None
        if client_ent:
            client_party = ClientParty(
                name=client_ent.text,
                nip=client_nip,
                name_span=(client_ent.start, client_ent.end),
                nip_span=(client_nip_ent.start, client_nip_ent.end) if client_nip_ent else None,
                score=client_ent.score,
                is_nip_valid=is_client_nip_valid,
                raw_nip_text=text[client_nip_ent.start : client_nip_ent.end]
                if client_nip_ent
                else None,
            )

        # Audit validation and Human-in-the-loop review recommendations
        needs_review = False
        review_reasons: list[str] = []

        if is_broker_nip_valid is False:
            needs_review = True
            review_reasons.append("Niepoprawna suma kontrolna Modulo 11 dla NIP brokera")

        if is_client_nip_valid is False:
            needs_review = True
            review_reasons.append("Niepoprawna suma kontrolna Modulo 11 dla NIP klienta")

        if broker_party is None:
            needs_review = True
            review_reasons.append("Nie zidentyfikowano brokera w dokumencie")
        elif broker_party.nip is None:
            review_reasons.append("Brak przypisanego numeru NIP dla brokera")

        if client_party is None:
            needs_review = True
            review_reasons.append("Nie zidentyfikowano klienta w dokumencie")
        elif client_party.nip is None:
            review_reasons.append("Brak przypisanego numeru NIP dla klienta")

        companies = [e for e in entities if e.label in ("company", "broker", "client")]
        if len(companies) >= 2:
            scored = []
            for c in companies[:2]:
                b_s, c_s = self._score_company(c, text)
                scored.append(b_s - c_s)
            margin = abs(scored[0] - scored[1])
            if margin < 1.5:
                needs_review = True
                review_reasons.append(
                    f"Niski margines pewności rozstrzygnięcia ról stron (margines={margin:.1f})"
                )

        overall_conf = 1.0 if not needs_review else 0.75

        return BrokerClientExtraction(
            broker=broker_party,
            client=client_party,
            confidence=overall_conf,
            needs_review=needs_review,
            review_reasons=review_reasons,
        )
