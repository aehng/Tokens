from __future__ import annotations

import unittest

from src.zip2zip.predictor_v2.attribution_reanalysis import (
    analyze_records,
    codebook_opportunity,
    count_phrase_occurrences,
    live_realization,
    oracle_ceiling,
)


def _record(condition: str, tokens: list[int], *, phrases=None, emissions=None, split="DEV"):
    phrases = phrases or []
    emissions = emissions or []
    return {
        "schema": "phi_attribution_record_v1",
        "prompt_id": "tiny-1",
        "domain": "code",
        "split": split,
        "condition": condition,
        "selected_h_slots": [list(phrase) for phrase in phrases],
        "generated_token_ids": list(tokens),
        "expanded_token_ids": list(tokens),
        "generated_token_count": len(tokens),
        "expanded_token_count": len(tokens),
        "transformer_decode_calls": len(tokens),
        "h_emissions": emissions,
        "quality_gate_pass": True,
    }


class TestPhiAttributionReanalysis(unittest.TestCase):
    def test_overlapping_occurrences_and_dp_savings_are_distinct(self):
        stats = count_phrase_occurrences([1001, 1001, 1001, 1001], [(1001, 1001)])
        self.assertEqual(stats["occurrence_count_overlapping"], 3)
        opportunity = codebook_opportunity([1001, 1001, 1001, 1001], [(1001, 1001)])
        self.assertEqual(opportunity["tokens_saved"], 2)
        self.assertEqual(opportunity["optimal_h_substitutions"], 2)

    def test_zero_opportunity_when_no_supplied_phrase_occurs(self):
        opportunity = codebook_opportunity([1, 2, 3], [(4, 5)])
        self.assertEqual(opportunity["tokens_saved"], 0)
        self.assertEqual(opportunity["supplied_phrases_that_occur"], 0)
        self.assertEqual(opportunity["optimal_h_substitutions"], 0)

    def test_oracle_ceiling_respects_k_and_phrase_lengths_and_is_deterministic(self):
        tokens = [1001, 1002, 1003, 1004, 1001, 1002, 1003, 1004, 1009, 1001, 1002, 1003, 1004]
        first = oracle_ceiling(tokens, k=2, min_length=2, max_length=4)
        second = oracle_ceiling(tokens, k=2, min_length=2, max_length=4)
        self.assertLessEqual(first["codebook_size"], 2)
        self.assertTrue(all(2 <= len(phrase) <= 4 for phrase in first["selected_phrases"]))
        self.assertEqual(first["selected_phrases"], second["selected_phrases"])
        self.assertEqual(first["tokens_saved"], second["tokens_saved"])

    def test_live_savings_reconcile_to_emission_spans_and_exact_slots(self):
        phrase = [1007, 1008, 1009]
        record = _record(
            "C_oracle",
            [32011, 99, 32007],
            phrases=[phrase],
            emissions=[{"pos": 0, "id": 32011, "subtokens": phrase}],
        )
        record["expanded_token_ids"] = [1007, 1008, 1009, 99, 32007]
        record["expanded_token_count"] = 5
        measured = live_realization(record)
        self.assertEqual(measured["realized_tokens_saved"], 2)
        self.assertEqual(measured["h_emissions"], 1)
        self.assertEqual(measured["mean_h_span_length"], 3.0)

    def test_reanalysis_rejects_final_records(self):
        records = [_record(condition, [1, 2, 32007], split="FINAL") for condition in (
            "A_vanilla", "B_h_disabled", "C_oracle", "D_real_predictor"
        )]
        with self.assertRaisesRegex(ValueError, "DEV records only"):
            analyze_records(records)

    def test_full_matrix_reconstructs_c_slots_and_keeps_categories_separate(self):
        seq = [1001, 1002, 1001, 1002, 1001, 1002, 32007]
        a = _record("A_vanilla", seq)
        # The expected C phrases are obtained by asking the public reanalysis to
        # use the same A-derived C selection rule represented in the live record.
        from src.zip2zip.predictor_v2.attribution_reanalysis import _derive_stage1_c_codebook
        c_phrases = _derive_stage1_c_codebook(seq)
        c = _record("C_oracle", seq, phrases=c_phrases)
        b = _record("B_h_disabled", [1001, 1002, 1001, 1003, 1004, 1005, 32007])
        d = _record("D_real_predictor", seq, phrases=[(1001, 1002)])
        report = analyze_records([a, b, c, d])
        self.assertTrue(report["per_prompt"][0]["c_codebook_reconstruction_matches_stage1_builder_on_a"])
        self.assertIn("oracle_a_ceiling", report["aggregate"])
        self.assertIn("c_exact_supplied_codebook_opportunity", report["aggregate"])
        self.assertIn("c_live_realization", report["aggregate"])
        self.assertEqual(report["methodology"]["final_accessed"], False)


if __name__ == "__main__":
    unittest.main()
