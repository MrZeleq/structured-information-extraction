#!/usr/bin/env python3
"""Example demonstrating extraction of Broker, Client, and their NIPs from Polish documents."""

import json

from structured_ie.pipeline.broker_client_pipeline import BrokerClientPipeline


def main() -> None:
    print("=" * 70)
    print("BROKER & CLIENT PRODUCTION EXTRACTION DEMO")
    print("=" * 70)

    # Sample realistic Polish brokerage documents:
    # 1. Standard authorization
    # 2. Long registration block (KRS, REGON, address 200+ chars before NIP)
    # 3. Boutique broker without brand root + OCR label typo (N1P)
    sample_texts = [
        (
            "Kancelaria Brokerska Nord Partner Sp. z o.o. (NIP: 5841234567), działając jako broker"
            " ubezpieczeniowy na rzecz klienta Alfa Logistyka S.A. (NIP: 5259876543), składa"
            " zapytanie ofertowe."
        ),
        (
            "Mocodawca Fabryka Maszyn Pol-Tech Sp. z o.o. z siedzibą w Warszawie przy ul. Prostej 10, "
            "00-123 Warszawa, wpisana do rejestru przedsiębiorców KRS pod numerem 0000123456, "
            "REGON 123456789, posługująca się numerem NIP: 5250007738, upoważnia dom brokerski "
            "Kancelaria Brokerska Lex Sp. z o.o., NIP: 5260251049."
        ),
        (
            "Zleceniodawca Budimex-Trans Sp. z o.o. o numerze N1P: 525 000 77 38 powierza obsługę "
            "brokerską spółce Doradztwo Finansowe Omega Sp. z o.o. (NIP: 526-025-10-49)."
        ),
    ]

    print("Initializing BrokerClientPipeline...")
    pipeline = BrokerClientPipeline.from_pretrained()

    for idx, text in enumerate(sample_texts, 1):
        print(f"\n--- Document {idx} ---")
        print(f"Text: {text}\n")

        # 1. High-level structured extraction (Business Model with Modulo 11 and Audit info)
        structured = pipeline.extract_structured(text)
        print("Structured Business Result:")
        print(json.dumps(structured.to_dict(), ensure_ascii=False, indent=2))

        # 2. Standard relational extraction (Entities + Graph Relations)
        res = pipeline.extract(text)
        print("\nExtracted Entities:")
        for e in res.entities:
            print(f"  [{e.label:7s}] '{e.text}' (span: {e.start}:{e.end})")

        print("Extracted Relations:")
        for r in res.relations:
            print(f"  ({r.source}) --[{r.relation}]--> ({r.target})")


if __name__ == "__main__":
    main()
