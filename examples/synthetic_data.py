"""Synthetic business datasets for demonstrating and testing zero-shot NER and Relation Extraction.

All companies, identifiers, policy numbers, and persons are completely synthetic.
"""

from structured_ie.schemas.models import Entity, LabeledSample, Relation

SYNTHETIC_ENTITY_LABELS = [
    "company",
    "person",
    "NIP",
    "policy number",
]

SYNTHETIC_RELATION_LABELS = [
    "works for",
    "has NIP",
    "broker for",
    "insured by",
    "has policy",
    "contact person",
]


def _build_entity(text: str, full_text: str, label: str) -> Entity:
    """Helper to find exact character start and end offsets."""
    start = full_text.index(text)
    end = start + len(text)
    return Entity(text=text, label=label, score=1.0, start=start, end=end)


def get_synthetic_samples() -> list[LabeledSample]:
    """Return 5 synthetic business scenarios for experiments and testing."""

    # -------------------------------------------------------------
    # Example 1: Company with NIP and Employee
    # -------------------------------------------------------------
    text1 = (
        "Alfa Logistyka S.A. o numerze NIP 5252525252 zatrudnia Tomasza Kamińskiego "
        "na stanowisku głównego dyspozytora."
    )
    e1_company = _build_entity("Alfa Logistyka S.A.", text1, "company")
    e1_nip = _build_entity("5252525252", text1, "NIP")
    e1_person = _build_entity("Tomasza Kamińskiego", text1, "person")

    sample1 = LabeledSample(
        text=text1,
        expected_entities=[e1_company, e1_nip, e1_person],
        expected_relations=[
            Relation(
                source="Alfa Logistyka S.A.",
                relation="has NIP",
                target="5252525252",
                score=1.0,
            ),
            Relation(
                source="Tomasza Kamińskiego",
                relation="works for",
                target="Alfa Logistyka S.A.",
                score=1.0,
            ),
        ],
        metadata={
            "description": "Example 1: Company with NIP identifier and employee.",
            "difficulty": "standard",
        },
    )

    # -------------------------------------------------------------
    # Example 2: Client and Broker Relationship
    # -------------------------------------------------------------
    text2 = (
        "Piotr Zieliński reprezentujący Beta Trade sp. z o.o. podpisał umowę o obsługę "
        "brokerską z firmą Omega Broker Sp. k."
    )
    e2_person = _build_entity("Piotr Zieliński", text2, "person")
    e2_client = _build_entity("Beta Trade sp. z o.o.", text2, "company")
    e2_broker = _build_entity("Omega Broker Sp. k.", text2, "company")

    sample2 = LabeledSample(
        text=text2,
        expected_entities=[e2_person, e2_client, e2_broker],
        expected_relations=[
            Relation(
                source="Piotr Zieliński",
                relation="works for",
                target="Beta Trade sp. z o.o.",
                score=1.0,
            ),
            Relation(
                source="Omega Broker Sp. k.",
                relation="broker for",
                target="Beta Trade sp. z o.o.",
                score=1.0,
            ),
        ],
        metadata={
            "description": "Example 2: Client and broker relationship with representative.",
            "difficulty": "standard",
        },
    )

    # -------------------------------------------------------------
    # Example 3: Company, Insurer, Broker and Policy Number
    # -------------------------------------------------------------
    text3 = (
        "Broker Quantum Re wynegocjował dla Gamma Tech S.A. polisę ubezpieczeniową "
        "POL-2024-99812 w towarzystwie Asekuracja Direct."
    )
    e3_broker = _build_entity("Quantum Re", text3, "company")
    e3_client = _build_entity("Gamma Tech S.A.", text3, "company")
    e3_policy = _build_entity("POL-2024-99812", text3, "policy number")
    e3_insurer = _build_entity("Asekuracja Direct", text3, "company")

    sample3 = LabeledSample(
        text=text3,
        expected_entities=[e3_broker, e3_client, e3_policy, e3_insurer],
        expected_relations=[
            Relation(
                source="Quantum Re",
                relation="broker for",
                target="Gamma Tech S.A.",
                score=1.0,
            ),
            Relation(
                source="Gamma Tech S.A.",
                relation="insured by",
                target="Asekuracja Direct",
                score=1.0,
            ),
            Relation(
                source="Gamma Tech S.A.",
                relation="has policy",
                target="POL-2024-99812",
                score=1.0,
            ),
        ],
        metadata={
            "description": "Example 3: Multi-role insurance scenario (broker, client, insurer, policy).",
            "difficulty": "moderate",
        },
    )

    # -------------------------------------------------------------
    # Example 4: Multiple Companies and Identifiers
    # -------------------------------------------------------------
    text4 = (
        "Konsorcjum tworzone przez Delta Motors z NIP 1112223344 oraz Epsilon Trans z NIP "
        "5556667788 wyznaczyło Annę Nowak na koordynatora projektu."
    )
    e4_c1 = _build_entity("Delta Motors", text4, "company")
    e4_nip1 = _build_entity("1112223344", text4, "NIP")
    e4_c2 = _build_entity("Epsilon Trans", text4, "company")
    e4_nip2 = _build_entity("5556667788", text4, "NIP")
    e4_person = _build_entity("Annę Nowak", text4, "person")

    sample4 = LabeledSample(
        text=text4,
        expected_entities=[e4_c1, e4_nip1, e4_c2, e4_nip2, e4_person],
        expected_relations=[
            Relation(
                source="Delta Motors",
                relation="has NIP",
                target="1112223344",
                score=1.0,
            ),
            Relation(
                source="Epsilon Trans",
                relation="has NIP",
                target="5556667788",
                score=1.0,
            ),
            Relation(
                source="Annę Nowak",
                relation="works for",
                target="Delta Motors",
                score=1.0,
            ),
        ],
        metadata={
            "description": "Example 4: Multiple co-occurring companies and respective NIPs.",
            "difficulty": "moderate",
        },
    )

    # -------------------------------------------------------------
    # Example 5: Intentionally Ambiguous Scenario
    # -------------------------------------------------------------
    text5 = (
        "Jan Kowalski rozmawiał z doradcami Zeta Broker oraz holdingu Nova Capital o umowie, "
        "jednak to Marek Wiśniewski został wskazany jako ostateczny decydent."
    )
    e5_p1 = _build_entity("Jan Kowalski", text5, "person")
    e5_c1 = _build_entity("Zeta Broker", text5, "company")
    e5_c2 = _build_entity("Nova Capital", text5, "company")
    e5_p2 = _build_entity("Marek Wiśniewski", text5, "person")

    sample5 = LabeledSample(
        text=text5,
        expected_entities=[e5_p1, e5_c1, e5_c2, e5_p2],
        expected_relations=[],  # Intentionally ambiguous — no definitive syntactic relationship
        metadata={
            "description": "Example 5: Ambiguous context where NER succeeds but relation extraction lacks clear evidence.",
            "difficulty": "hard",
        },
    )

    return [sample1, sample2, sample3, sample4, sample5]
