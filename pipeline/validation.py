import json
import re
from enum import Enum
from os.path import join
from typing import List

from genson import SchemaBuilder

from utils.config import AppConfig


class ValidationError(Enum):
    """
    Possible validation errors in the produced json
    """
    empty_topics = "empty-topics"
    empty_relations = "empty-relations"
    empty_adus = "empty-adus"
    empty_major_claims = "empty-major-claims"
    empty_claims = "empty-claims"
    empty_premises = "empty-premises"
    claim_without_stance = "claim_without_stance"
    source_premise_target_claim_invalid = "source-premise-target-claim-invalid"
    source_claim_target_major_claim_invalid = "source-premise-target-claim-invalid"
    premise_source_invalid = "premise-source-invalid"
    major_claim_target_invalid = "major-claim-target-invalid"
    relation_confidence_empty = "relation-confidence-empty"
    missing_adus = "missing-adus"
    major_claim_missing_relations = "major-claim-missing-relations"
    claims_missing_relations_source = "claims-missing-relations-source"
    claims_missing_relations_target = "claims-missing-relations-target"
    premises_missing_relations = "premises-missing-relations"


class JsonValidator:

    def __init__(self, app_config: AppConfig):
        """
        Constructor for JsonValidator class

        Args
            app_config (AppConfig): configuration parameters
        """
        self.app_config = app_config
        self.app_logger = app_config.app_logger

    def validate(self, document):
        """
        Gets a document (in json format) and validates it based on specific rules:
        |    1. topics list must not be empty
        |    2. relations list must not be empty
        |    3. ADUs list must not be empty
        |    4. All ADU types should be present, i.e. major_claim, claim, premise
        |    5. All ADU segments should be associated in a relation

        Args
            document (dict): the document in json format to be validated

        Returns
            list: a list of validation errors found in the document - if the document is valid, the list is empty
        """
        self.app_logger.info(f"Validating document with id {document['id']} and title {document['title']}")
        validation_errors, invalid_adus = [], []
        # check topics, relations and ADUs lists are not empty
        if not document["topics"]:
            self.app_logger.warning("Document does not contain topics")
            validation_errors.append(ValidationError.empty_topics)
        if not document["annotations"]["Relations"]:
            self.app_logger.warning("Document has an empty relations list")
            validation_errors.append(ValidationError.empty_relations)
        if not document["annotations"]["ADUs"]:
            self.app_logger.warning("Document has empty ADUs list")
            validation_errors.append(ValidationError.empty_adus)

        # if relations or ADUs are empty there is no need to continue the following validations
        if ValidationError.empty_relations in validation_errors or ValidationError.empty_adus in validation_errors:
            self.app_logger.info("Relations or ADUs are empty - Stopping validation")
            return validation_errors, invalid_adus

        adus = document["annotations"]["ADUs"]
        relations = document["annotations"]["Relations"]

        # check major claim, claim and premise types exist
        self.app_logger.debug("Collecting major claim, claims and premises")
        major_claims, claims, premises = [], [], []
        for adu in adus:
            adu_type = adu["type"]
            if adu_type == "major_claim":
                major_claims.append(adu)
            if adu_type == "claim":
                claims.append(adu)
            if adu_type == "premise":
                premises.append(adu)

        if not major_claims or not claims or not premises:
            # if any type is missing, no need to check for relations
            if not major_claims:
                self.app_logger.warning("The document does not contain major claim - Stopping validation")
                validation_errors.append(ValidationError.empty_major_claims)
            if not claims:
                self.app_logger.warning("The document does not contain any claims - Stopping validation")
                validation_errors.append(ValidationError.empty_claims)
            if not premises:
                self.app_logger.warning("The document does not contain any premises - Stopping validation")
                validation_errors.append(ValidationError.empty_premises)
            return validation_errors, invalid_adus

        val_errors, invalid_claims = self._validate_stance(claims=claims)
        validation_errors += val_errors
        validation_errors += self._validate_relations(relations=relations, adus=adus)

        self.app_logger.info("Checking if all ADUs are present in the relations list")
        major_claims_rel, invalid_major_claims = self._relation_exists(relations=relations, adus=major_claims,
                                                                       position="target")
        claims_rel_source, invalid_claims_rel_source = self._relation_exists(relations=relations, adus=claims,
                                                                             position="source")
        claims_rel_target, invalid_claims_rel_target = self._relation_exists(relations=relations, adus=claims,
                                                                             position="target")
        premises_rel, invalid_premises = self._relation_exists(relations=relations, adus=premises, position="source")
        invalid_adus = invalid_claims + invalid_major_claims + invalid_claims_rel_source + invalid_claims_rel_target
        invalid_adus += invalid_premises
        if not major_claims_rel or (not claims_rel_target and not claims_rel_source) or not premises_rel:
            if not major_claims_rel:
                self.app_logger.warning("Missing relations for major claim")
                validation_errors.append(ValidationError.major_claim_missing_relations)
            if not claims_rel_source:
                self.app_logger.warning("Missing relations for some claims towards the major claim")
                validation_errors.append(ValidationError.claims_missing_relations_source)
            if not claims_rel_target:
                self.app_logger.warning("Missing relations for some relations of some claims with premises")
                validation_errors.append(ValidationError.claims_missing_relations_target)
            if not premises_rel:
                self.app_logger.warning("Missing relations for some premises")
                validation_errors.append(ValidationError.premises_missing_relations)
        self.app_logger.info(f"Validation finished! Found {len(validation_errors)} errors")
        return validation_errors, invalid_adus

    def export_json_schema(self, document_ids):
        """
        Extracts for some given documents, their json schema and saves it into a file

        Args
            documents_ids (list): the ids of the documents in the elasticsearch
        """
        res = self.app_config.elastic_save.elasticsearch_client.mget(index="debatelab", body={"ids": document_ids})
        res = res["docs"]
        builder = SchemaBuilder()
        builder.add_schema({"type": "object", "properties": {}})
        for doc in res:
            builder.add_object(doc)
        schema = builder.to_json()
        file_path = join(self.app_config.output_path, "schema.json")
        with open(file_path, "w") as f:
            f.write(json.dumps(schema, indent=4, sort_keys=False))

    def _validate_relations(self, relations, adus):
        self.app_logger.info("Validation of relations regarding the ADU positions & the existence of confidence")
        validation_errors = []
        idx = 0
        while idx < len(relations):
            relation = relations[idx]
            source = relation["arg1"]
            target = relation["arg2"]
            confidence = relation.get("confidence", None)
            for adu in adus:
                if adu["id"] == source:
                    source = adu
                elif adu["id"] == target:
                    target = adu
            if type(target) != dict or type(source) != dict:
                validation_errors.append(ValidationError.missing_adus)
            if target["type"] == "premise":
                validation_errors.append(ValidationError.premise_source_invalid)
            if source["type"] == "major_claim":
                validation_errors.append(ValidationError.major_claim_target_invalid)
            if not confidence:
                validation_errors.append(ValidationError.relation_confidence_empty)
            if (source["type"] == "premise" and target["type"] != "claim") or (
                    target["type"] == "claim" and source["type"] != "premise"):
                validation_errors.append(ValidationError.source_premise_target_claim_invalid)
            if (source["type"] == "claim" and target["type"] != "major_claim") or (
                    target["type"] == "major_claim" and source["type"] != "claim"):
                validation_errors.append(ValidationError.source_claim_target_major_claim_invalid)
            if validation_errors:
                break
            else:
                idx += 1
        return validation_errors

    def _validate_stance(self, claims):
        self.app_logger.info("Claim validation for stance")
        validation_errors = []
        invalid_claims = []
        found_invalid = False
        for claim in claims:
            stance = claim.get("stance", None)
            if not stance:
                if not found_invalid:
                    validation_errors.append(ValidationError.claim_without_stance)
                    found_invalid = True
                invalid_claims.append(claim)
        return validation_errors, invalid_claims

    @staticmethod
    def _relation_exists(relations, adus, position):
        """
        For each ADU (source or target) find if there are any relations

        Args
            | relations (list): a list of all the predicted relations | adus (list): list of all the predicted ADUs
            | position (str): string with values source or target indicating the ADU position

        Returns
            bool: True/False based on whether all ADUs are present in the Relations list
        """
        found_relations = []
        invalid_adus = []
        for adu in adus:
            found = False
            for relation in relations:
                arg_id = relation["arg1"] if position == "source" else relation["arg2"]
                if arg_id == adu["id"]:
                    found_relations.append(relation)
                    found = True
                if not found:
                    invalid_adus.append(adu)
        flag = True if not invalid_adus else False
        return flag, invalid_adus


class JsonCorrector:
    """
    Component that performs corrections based on the validation errors
    """

    def __init__(self, app_config: AppConfig, segment_counter: int, rel_counter: int, stance_counter: int):
        """
        Constructor for the JsonCorrector class

        Args
            | app_config (AppConfig): the configuration parameters of the application
            | segment_counter (int): the counter for ADUs
            | rel_counter (int): the counter of relations in the list
            | stance_counter (int): the counter of stance in the relevant list
        """
        self.app_config = app_config
        self.app_logger = app_config.app_logger
        self.segment_counter = segment_counter
        self.rel_counter = rel_counter
        self.stance_counter = stance_counter

    @staticmethod
    def can_document_be_corrected(validation_errors: List[ValidationError]):
        """
        Function to check if a document can be corrected. If there are errors such as empty lists - topics, relations,
        ADUs - or empty ADU types (no major claim, claim or premise), then the document cannot be corrected. However,
        if an ADU does not have any relation, the document can be corrected and the ADU without relations should be
        removed

        Args
            validation_errors (list): validation errors for a specific document

        Returns
            bool: True/False indicating if the document can be corrected
        """
        unaccepted_errors = [ValidationError.empty_topics, ValidationError.empty_adus, ValidationError.empty_relations,
                             ValidationError.empty_major_claims, ValidationError.empty_claims,
                             ValidationError.empty_premises, ValidationError.missing_adus,
                             ValidationError.source_claim_target_major_claim_invalid,
                             ValidationError.source_premise_target_claim_invalid,
                             ValidationError.premise_source_invalid, ValidationError.relation_confidence_empty,
                             ValidationError.major_claim_target_invalid]
        return False if any(error in validation_errors for error in unaccepted_errors) else True

    def correction(self, document):
        """
        Function to perform corrections to a document - only to the documents that the function
        ```can_document_be_corrected()``` returned True

        Args
            document (dict): the document in json format to be corrected

        Returns
            dict: the corrected document
        """
        adus = document["annotations"]["ADUs"]
        relations = document["annotations"]["Relations"]
        major_claims = [adu for adu in adus if adu["type"] == "major_claim"]
        claims = [adu for adu in adus if adu["type"] == "claim"]
        premises = [adu for adu in adus if adu["type"] == "premise"]
        # check if major claim is split - predictions on sentence level
        if len(major_claims) > 1:
            adus = document["annotations"]["ADUs"]
            adus = self.handle_multiple_major_claims(adus=adus, major_claims=major_claims)
            major_claims = [adu for adu in adus if adu["type"] == "major_claim"]
        claims, relations = self.update_claims_with_relations(claims=claims, relations=relations,
                                                              major_claim=major_claims[0])
        premises = self.update_premises_with_relations(premises=premises, relations=relations)
        adus = major_claims + claims + premises
        document["annotations"]["ADUs"] = adus
        document["annotations"]["Relations"] = relations
        return document

    @staticmethod
    def handle_multiple_major_claims(adus, major_claims):
        """
        Concatenates the text of major claims (predictions were on sentence-level)

        Args
            | adus (list): a list of all ADUs of the document
            | major_claims (list): a list with the major claims (split into multiple segments)

        Returns
            list: updated list of ADUs with one major claim
        """
        major_claim_txt = " ".join([major_claim["segment"] for major_claim in major_claims])
        re.sub(' +', ' ', major_claim_txt)
        major_claim = major_claims[0]
        major_claim["segment"] = major_claim_txt
        major_claim["ends"] = major_claim["starts"] + len(major_claim_txt)
        new_adus = [major_claim]
        for adu in adus:
            if adu["type"] == "major_claim":
                continue
            new_adus.append(adu)
        return new_adus

    def update_premises_with_relations(self, premises, relations):
        """
        Function to keep only the premises that have relations with claims

        Args
            | premises (list): a list of the premises
            | relations (list): a list with the predicted relations

        Returns
            list: a list with the premises to be kept
        """
        new_premises = []
        for premise in premises:
            premise_rel = self._get_adu_relations(adu_id=premise["id"], relations=relations, position="source")
            if premise_rel is not None and len(premise_rel) > 0:
                new_premises.append(premise)
            else:
                self.app_logger.warning(
                    f"Missing relation for premise with id {premise['id']} and text {premise['segment']}")
        return new_premises

    def update_claims_with_relations(self, claims, relations, major_claim):
        """
        Get the updated list of claims and relations. Claims kept are those that have stance towards the major claim
        and have at least one relation where they are the source ADU. In case that a claim does not have a stance
        and there is no relation with the claim as source, but there are relations with the claim as target, the
        relevant relations are removed.

        Args
            | claims (list): a list with all the predicted claims
            | relations (list): a list with all the predicted relations
            | major_claim (dict): the major claim of the document

        Returns
            tuple: the updated lists of claims and relations
        """
        new_claims = []
        for claim in claims:
            source_relations = self._get_adu_relations(adu_id=claim["id"], relations=relations,
                                                       position="source")
            target_relations = self._get_adu_relations(adu_id=claim["id"], relations=relations,
                                                       position="target")
            stance = claim.get("stance", [])
            source_rel_exists = True if source_relations is not None and len(source_relations) > 0 else False
            target_rel_exists = True if target_relations is not None and len(target_relations) > 0 else False
            if stance and not source_rel_exists:
                stance_type = stance[0]["type"]
                rel_type = "support" if stance_type == "for" else "attack"
                self.rel_counter += 1
                relation = {
                    "id": f"R{self.rel_counter}",
                    "type": rel_type,
                    "arg1": claim["id"],
                    "arg2": major_claim["id"],
                    "confidence": stance[0]["confidence"]
                }
                relations.append(relation)
                source_rel_exists = True
            elif source_rel_exists and not stance:
                self.stance_counter += 1
                rel_type, confidence = None, None
                for relation in relations:
                    if relation["arg1"] == claim["id"] and relation["arg2"] == major_claim["id"]:
                        rel_type = relation["type"]
                        confidence = relation["confidence"]
                if rel_type is not None and rel_type != "":
                    stance = {
                        "id": f"A{self.stance_counter}",
                        "type": rel_type,
                        "confidence": confidence
                    }
                    claim["stance"] = [stance]
            stance = claim.get("stance", [])
            if stance and source_rel_exists:
                new_claims.append(claim)
            else:
                if target_rel_exists:
                    for target_relation in target_relations:
                        relations.remove(target_relation)
        return new_claims, relations

    @staticmethod
    def _get_adu_relations(adu_id, relations, position):
        """
        Based on an ADU and the position (source or target), the function searches for associated relations

        Args
            | adu_id (str): the id of the ADU
            | relations (list): the list of all the predicted relations
            | position (str): valid values are source and target

        Returns
            list: relations associated with the ADU and the position requested
        """
        relations_found = []
        for relation in relations:
            arg_id = relation["arg1"] if position == "source" else relation["arg2"]
            if arg_id == adu_id:
                relations_found.append(relation)
        return relations_found
