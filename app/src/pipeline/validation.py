import json
from datetime import datetime
from enum import Enum
from os.path import join
from typing import List

from genson import SchemaBuilder

from src.utils.config import AppConfig


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

    def run_validation(self, document: dict, counters: dict, do_correction: bool = True):
        validation_errors, invalid_adus = self.validate(document=document)
        corrected = False
        if do_correction and validation_errors:
            corrected = True
            counter = self.app_config.properties["eval"]["max_correction_tries"]
            corrector = JsonCorrector(app_config=self.app_config, counters=counters)
            while counter > 0 and validation_errors:
                if not corrector.can_document_be_corrected(validation_errors=validation_errors):
                    break
                document = corrector.correction(document=document, invalid_adus=invalid_adus)
                validation_errors, invalid_adus = self.validate(document=document)
                counter -= 1
        return validation_errors, invalid_adus, corrected

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
        validation_errors, invalid_adus = [], {}
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
        invalid_adus["stance"] = invalid_claims
        validation_errors += val_errors
        validation_errors += self._validate_relations(relations=relations, adus=adus)

        self.app_logger.info("Checking if all ADUs are present in the relations list")
        major_claims_rel, invalid_major_claims = self._relation_exists(relations=relations, adus=major_claims,
                                                                       position="target")
        invalid_adus["major_claim"] = invalid_major_claims
        claims_rel_source, invalid_claims_rel_source = self._relation_exists(relations=relations, adus=claims,
                                                                             position="source")
        invalid_adus["claims_source"] = invalid_claims_rel_source
        premises_rel, invalid_premises = self._relation_exists(relations=relations, adus=premises, position="source")
        invalid_adus["premises"] = invalid_premises
        if not major_claims_rel or not claims_rel_source or not premises_rel:
            if not major_claims_rel:
                self.app_logger.warning("Missing relations for major claim")
                validation_errors.append(ValidationError.major_claim_missing_relations)
            if not claims_rel_source:
                self.app_logger.warning("Missing relations for some claims towards the major claim")
                validation_errors.append(ValidationError.claims_missing_relations_source)
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
            else:
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
        invalid_adus = []
        for adu in adus:
            found = False
            for relation in relations:
                arg_id = relation["arg1"] if position == "source" else relation["arg2"]
                if arg_id == adu["id"]:
                    found = True
            if not found:
                invalid_adus.append(adu)
        flag = True if not invalid_adus else False
        return flag, invalid_adus

    def save_invalid_json(self, document, validation_errors, invalid_adus):
        """
        Function to save an invalid json object into the output_files directory

        Args
            document (dict)
        """
        self.app_logger.debug("Writing invalid document into file")
        timestamp = datetime.now()
        filename = f"{document['id']}_{timestamp}.json"
        file_path = join(self.app_config.output_files, filename)
        with open(file_path, "w", encoding='utf8') as f:
            f.write(json.dumps(document, indent=4, sort_keys=False, ensure_ascii=False))
        with open(f"{file_path}.txt", "w") as f:
            for validation_error in validation_errors:
                f.write(validation_error.value + "\n")
            f.write(str(invalid_adus) + "\n")

    def print_validation_results(self, document_ids, corrected_ids, invalid_document_ids, print_corrected=False):
        self.app_logger.info(f"Total valid documents: {len(document_ids)}")
        if print_corrected:
            self.app_logger.info(f"Total corrected documents: {len(corrected_ids)}")
        self.app_logger.info(f"Total invalid documents: {len(invalid_document_ids)}")
        self.app_logger.warning(f"Invalid document ids: {invalid_document_ids}")


class JsonCorrector:
    """
    Component that performs corrections based on the validation errors
    """

    def __init__(self, app_config: AppConfig, counters: dict):
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
        self.segment_counter = counters["adu"]
        self.rel_counter = counters["rel"]
        self.stance_counter = counters["stance"]

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

    def correction(self, document, invalid_adus):
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

        # fix stance
        document = self.handle_claims_without_stance(document=document, claims=invalid_adus["stance"],
                                                     relations=relations)
        # fix claims as source in relations
        document = self.handle_source_missing_claims(document=document, claims=invalid_adus["claims_source"],
                                                     major_claim=major_claims[0])
        # claims
        # handle missing premises
        document = self.handle_missing_premises(document=document, premises=invalid_adus["premises"])
        return document

    def handle_claims_without_stance(self, document, claims, relations):
        """
        Gets the invalid claims, that do not have stance, and checks the relations list. If a relation exists between
        a claim without stance and its major claim, then the stance is updated based on the relation and thus the error
        of missing stance is fixed.

        Args
            | document (dict): the json object containing a specific document
            | claims (list): a list of invalid claims extracted from the validation step
            | relations (list): the list of relations to lookup for major claim - claim relation

        Returns
            dict: the updated document
        """
        for claim in claims:
            for rel in relations:
                # assuming that we have only one major claim
                if rel["arg1"] == claim["id"]:
                    rel_type = rel["type"]
                    rel_type = "for" if rel_type == "support" else "against"
                    confidence = rel["confidence"]
                    self.stance_counter += 1
                    stance = {
                        "id": f"A{self.stance_counter}",
                        "type": rel_type,
                        "confidence": confidence
                    }
                    for adu in document["annotations"]["ADUs"]:
                        if adu["id"] == claim["id"]:
                            adu["stance"] = [stance]
        return document

    def handle_source_missing_claims(self, document, claims, major_claim):
        """
        Function to handle the errors from missing relations of claims as sources (i.e. major claim / claim relations).
        This function performs the opposite operation from the ```handle_claims_without_stance()``` function. More
        specifically, for each invalid claim, its stance is checked (if exists) and a new relation is created based on
        the stance so as to fix the error of missing relations.

        Args
            document (dict): the json object containing a document
            claims (list): the list of invalid claims
            major_claim (dict): the major claim of the document -- assuming that there is only one major claim
        """
        for invalid_claim in claims:
            for adu in document["annotations"]["ADUs"]:
                if invalid_claim["id"] == adu["id"]:
                    stance = adu.get("stance", [])
                    if stance:
                        stance_type = stance[0]["type"]
                        confidence = stance[0]["confidence"]
                        rel_type = "support" if stance_type == "for" else "attack"
                        self.rel_counter += 1
                        relation = {
                            "id": f"R{self.rel_counter}",
                            "type": rel_type,
                            "arg1": adu["id"],
                            "arg2": major_claim["id"],
                            "confidence": confidence
                        }
                        document["annotations"]["Relations"].append(relation)
        return document

    def handle_missing_premises(self, document, premises):
        """
        Function to fix errors from missing relations of premises/claims. If a premise does not have any relations, it
        is removed from the ADU list

        Args
            | document (dict): the json object that contains the document
            | premises (list): list of invalid premises

        Returns
            dict: the updated document
        """
        adus = document["annotations"]["ADUs"]
        adus_to_be_kept = []
        invalid_premise_ids = [premise["id"] for premise in premises]
        for adu in adus:
            if adu["id"] not in invalid_premise_ids:
                adus_to_be_kept.append(adu)
            else:
                self.app_logger.warning(
                    f"Missing relation for premise with id {adu['id']} and text {adu['segment']}")
        document["annotations"]["ADUs"] = adus_to_be_kept
        return document
