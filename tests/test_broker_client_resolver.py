"""Unit tests for BrokerClientResolver and BrokerClientPipeline."""

from unittest.mock import MagicMock

from structured_ie.pipeline.broker_client_pipeline import BrokerClientPipeline
from structured_ie.postprocessing.broker_client_resolver import (
    BrokerClientExtraction,
    BrokerClientResolver,
    BrokerParty,
    ClientParty,
)
from structured_ie.schemas.models import Entity


def test_bind_nips_to_companies_parenthetical() -> None:
    text = "Firma Alfa (NIP: 1111111111) współpracuje z Broker Beta (NIP: 2222222222)."
    companies = [
        Entity(text="Firma Alfa", label="company", score=0.9, start=0, end=10),
        Entity(text="Broker Beta", label="company", score=0.9, start=44, end=55),
    ]
    nips = [
        Entity(text="1111111111", label="NIP", score=0.9, start=17, end=27),
        Entity(text="2222222222", label="NIP", score=0.9, start=62, end=72),
    ]

    resolver = BrokerClientResolver()
    binding = resolver.bind_nips_to_companies(companies=companies, nips=nips, text=text)

    assert binding["Firma Alfa"].text == "1111111111"
    assert binding["Broker Beta"].text == "2222222222"


def test_score_company_cues() -> None:
    text = "Kancelaria Brokerska Nord Partner działa na rzecz klienta Alfa Logistyka."
    comp_broker = Entity(text="Nord Partner", label="company", score=0.9, start=21, end=33)
    comp_client = Entity(text="Alfa Logistyka", label="company", score=0.9, start=58, end=72)

    resolver = BrokerClientResolver()
    b_score_1, c_score_1 = resolver._score_company(comp_broker, text)
    b_score_2, c_score_2 = resolver._score_company(comp_client, text)

    assert b_score_1 > c_score_1, "Nord Partner should have higher broker score"
    assert c_score_2 > b_score_2, "Alfa Logistyka should have higher client score"


def test_resolve_two_parties() -> None:
    text = (
        "Kancelaria Brokerska Nord Partner Sp. z o.o. (NIP: 5841234567), "
        "działając jako broker na rzecz klienta Alfa Logistyka S.A. (NIP: 5259876543)."
    )
    entities = [
        Entity(text="Nord Partner Sp. z o.o.", label="company", score=0.95, start=21, end=44),
        Entity(text="5841234567", label="NIP", score=0.98, start=51, end=61),
        Entity(text="Alfa Logistyka S.A.", label="company", score=0.92, start=95, end=114),
        Entity(text="5259876543", label="NIP", score=0.98, start=121, end=131),
    ]

    resolver = BrokerClientResolver()
    out_ents, out_rels = resolver.resolve(entities=entities, text=text)

    labels = {e.text: e.label for e in out_ents}
    assert labels["Nord Partner Sp. z o.o."] == "broker"
    assert labels["Alfa Logistyka S.A."] == "client"

    rel_map = {(r.source, r.relation): r.target for r in out_rels}
    assert rel_map[("Nord Partner Sp. z o.o.", "has NIP")] == "5841234567"
    assert rel_map[("Alfa Logistyka S.A.", "has NIP")] == "5259876543"
    assert rel_map[("Nord Partner Sp. z o.o.", "broker for")] == "Alfa Logistyka S.A."


def test_extract_structured() -> None:
    text = (
        "Ubezpieczający TechSoft Sp. z o.o. (NIP: 1112223344) upoważnia Mentor S.A. (NIP:"
        " 9560001234)."
    )
    entities = [
        Entity(text="TechSoft Sp. z o.o.", label="company", score=0.9, start=15, end=34),
        Entity(text="1112223344", label="NIP", score=0.9, start=41, end=51),
        Entity(text="Mentor S.A.", label="company", score=0.95, start=63, end=74),
        Entity(text="9560001234", label="NIP", score=0.9, start=81, end=91),
    ]

    resolver = BrokerClientResolver()
    structured: BrokerClientExtraction = resolver.extract_structured(entities=entities, text=text)

    assert isinstance(structured.broker, BrokerParty)
    assert structured.broker.name == "Mentor S.A."
    assert structured.broker.nip == "9560001234"

    assert isinstance(structured.client, ClientParty)
    assert structured.client.name == "TechSoft Sp. z o.o."
    assert structured.client.nip == "1112223344"

    d = structured.to_dict()
    assert d["broker"]["name"] == "Mentor S.A."
    assert d["client"]["nip"] == "1112223344"


def test_resolve_empty_and_single() -> None:
    resolver = BrokerClientResolver()
    ents_empty, rels_empty = resolver.resolve(entities=[], text="brak danych")
    assert ents_empty == []
    assert rels_empty == []

    comp = Entity(text="Solo Broker S.A.", label="company", score=0.9, start=0, end=16)
    ents_single, _ = resolver.resolve(entities=[comp], text="Solo Broker S.A.")
    assert len(ents_single) == 1
    assert ents_single[0].label == "broker"


def test_broker_client_pipeline_mock() -> None:
    mock_ner = MagicMock()
    mock_ner.extract_entities.return_value = [
        Entity(text="Broker Mentor", label="company", score=0.9, start=0, end=13),
        Entity(text="NIP: 1234567890", label="NIP", score=0.9, start=15, end=30),
    ]

    pipeline = BrokerClientPipeline(ner_extractor=mock_ner)
    result = pipeline.extract(text="Broker Mentor (NIP: 1234567890)")

    assert len(result.entities) == 2
    nip_ent = next(e for e in result.entities if e.label == "NIP")
    assert nip_ent.text == "1234567890"

    structured = pipeline.extract_structured(text="Broker Mentor (NIP: 1234567890)")
    assert structured.broker is not None
    assert structured.broker.nip == "1234567890"


def test_bind_nips_long_registration_block() -> None:
    # 200+ characters of address, KRS, REGON between company name and NIP
    text = (
        "Mocodawca Fabryka Maszyn Sp. z o.o. z siedzibą w Warszawie przy ul. Prostej 10, "
        "00-123 Warszawa, wpisana do rejestru przedsiębiorców KRS pod numerem 0000123456, "
        "REGON 123456789, posługująca się numerem NIP: 5250007738, upoważnia "
        "Kancelarię Brokerską Lex Sp. z o.o., NIP 5260251049."
    )
    comp1 = Entity(text="Fabryka Maszyn Sp. z o.o.", label="company", score=0.9, start=10, end=35)
    nip1 = Entity(text="5250007738", label="NIP", score=0.9, start=190, end=200)
    comp2 = Entity(
        text="Kancelarię Brokerską Lex Sp. z o.o.", label="company", score=0.9, start=213, end=248
    )
    nip2 = Entity(text="5260251049", label="NIP", score=0.9, start=254, end=264)

    resolver = BrokerClientResolver(max_registration_distance=350)
    binding = resolver.bind_nips_to_companies(
        companies=[comp1, comp2],
        nips=[nip1, nip2],
        text=text,
    )

    assert binding["Fabryka Maszyn Sp. z o.o."].text == "5250007738"
    assert binding["Kancelarię Brokerską Lex Sp. z o.o."].text == "5260251049"


def test_boutique_broker_without_brand_roots() -> None:
    # "Alfa Finanse" is not a known brand, but the sentence clearly identifies its role
    text = (
        "Zleceniodawca Firma Budowlana Sp. z o.o. (NIP: 5250007738) powierza obsługę "
        "brokerską spółce Alfa Finanse Sp. z o.o. (NIP: 5260251049)."
    )
    comp_client = Entity(
        text="Firma Budowlana Sp. z o.o.", label="company", score=0.9, start=14, end=40
    )
    nip_client = Entity(text="5250007738", label="NIP", score=0.9, start=47, end=57)
    comp_broker = Entity(
        text="Alfa Finanse Sp. z o.o.", label="company", score=0.9, start=92, end=115
    )
    nip_broker = Entity(text="5260251049", label="NIP", score=0.9, start=122, end=132)

    resolver = BrokerClientResolver()
    structured = resolver.extract_structured(
        entities=[comp_client, nip_client, comp_broker, nip_broker],
        text=text,
    )

    assert structured.broker is not None
    assert structured.broker.name == "Alfa Finanse Sp. z o.o."
    assert structured.broker.nip == "5260251049"
    assert structured.broker.is_nip_valid is True

    assert structured.client is not None
    assert structured.client.name == "Firma Budowlana Sp. z o.o."
    assert structured.client.nip == "5250007738"
    assert structured.client.is_nip_valid is True

    assert structured.needs_review is False


def test_invalid_nip_flags_review() -> None:
    text = (
        "Ubezpieczający Klient Sp. z o.o. (NIP: 1234567890) upoważnia dom brokerski "
        "Omega Sp. z o.o. (NIP: 5250007738)."
    )
    comp_c = Entity(text="Klient Sp. z o.o.", label="company", score=0.9, start=15, end=32)
    nip_c = Entity(text="1234567890", label="NIP", score=0.9, start=39, end=49)  # Invalid checksum
    comp_b = Entity(text="Omega Sp. z o.o.", label="company", score=0.9, start=76, end=92)
    nip_b = Entity(text="5250007738", label="NIP", score=0.9, start=99, end=109)  # Valid checksum

    resolver = BrokerClientResolver()
    structured = resolver.extract_structured(
        entities=[comp_c, nip_c, comp_b, nip_b],
        text=text,
    )

    assert structured.client.is_nip_valid is False
    assert structured.broker.is_nip_valid is True
    assert structured.needs_review is True
    assert any("Niepoprawna suma kontrolna" in r for r in structured.review_reasons)
