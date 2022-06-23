import logging

from src.utils import make_segment_id


def run(stance_model, adu_data):
    """
    Stance prediction pipeline. It produces pairs of claims/major claims to predict their stance.

    Args
        | major_claims (list): list of the major claims of the document
        | claims (list): list of the claims of the document
        | json_obj (dict): the document

    Returns
        tuple: the updated document and the stance counter used to create ids for each stance
    """
    results = []
    for idx, adus in enumerate(adu_data):
        logging.info(f"Extracting stance predictions for doc {idx + 1} / {len(adu_data)}")
        major_claims, claims = adus["major_claims"], adus["claims"]
        result = {}
        # double-for with majors and claims
        for m, major_claim in enumerate(major_claims):
            logging.debug(f"Major claim {m + 1}/{len(major_claims)}")
            for claim in claims:
                claim_text, mc_text = claim["segment"], major_claim["segment"]
                logging.debug(f"Predicting relation for claim / mc pair: {(claim_text, mc_text)}")
                score, label = stance_model.predict(claim_text, mc_text)

                if label and label != stance_model.get_null_label():
                    stance_list = [{
                        "id": make_segment_id(),
                        "type": label,
                        "confidence": score
                    }]
                    result[claim["id"]] = stance_list
        results.append(result)
    return results
