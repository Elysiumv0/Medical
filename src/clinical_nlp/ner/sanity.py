def run_sanity_check(expected_entities, predicted_entities):
    matches = 0
    for exp in expected_entities:
        for pred in predicted_entities:
            if exp["text"] == pred["text"] and exp["type"] == pred["type"]:
                matches += 1
                break
    return matches / len(expected_entities) if expected_entities else 0