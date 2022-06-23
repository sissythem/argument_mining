from api import Session, Collection, Document, Span, Attribute
from ellogon import tokeniser
import sys

session = Session(username="debatelab@ellogon.iit.demokritos.gr",
                  password="Up9F6AE2YN",
                  URL='https://vast.ellogon.org/');

def token_in_annotations(annotation_set, start, end):
    for ann in annotation_set:
        if ann.spansIncludeOffsets(start, end):
            return ann
    return None

def token_label(annotation_set, start, end):
    ann = token_in_annotations(annotation_set, start, end)
    if ann:
        return ann.attributeGet('type').value
    return 'O'

##
## Export the collection names specified in the rguments...
##
sentence_id = 0
annotators = [
  'Button_Annotator_neutral_argument_type_Generic',
  'Button_Annotator_neutral_argument_type_Mixed_Argument-Stance_Relations'
]
for i,collection_name in enumerate(sys.argv[1:]):
    col = session.collectionGetByName(collection_name)
    for doc in col.documents():
        ## Simulate how codemirror handles lines...
        text_cm = "\n".join(doc.text.splitlines())
    
        annotations = dict()
        annotation_set = []
        sp = 0
        ## Collect ADU annotations...
        for ann in doc.annotationsByType('argument'):
            if ann.annotator_id not in annotators:
                continue
            segment_cm = text_cm[int(ann.spans[0].start) : int(ann.spans[0].end)]
            assert segment_cm == ann.spans[0].segment
            annotations[ann._id] = ann
            annotation_set.append(ann)
        ## Get all relations, and make sure we have all ADUs...
        for ann in doc.annotationsByType('argument_relation'):
            if ann.annotator_id not in annotators:
                continue
            if len(ann.spans) == 0:
                #print(ann)
                ## Relations...
                ## Ensure that all annotations exist!
                arg1 = ann.attributeGet('arg1')
                arg2 = ann.attributeGet('arg2')
                #print(ann._id, arg1, arg2, arg1.value, arg1.value in annotations)
                assert arg1.value in annotations
                assert arg2.value in annotations
        for tokens, sent_start, sent_end in tokeniser.tokenise_spans(text_cm):
            prev_label = 'O'
            for token, start, end in tokens:
                label = token_label(annotation_set, start, end)
                if label == 'O':
                    prefix = ''; mark = 'N'
                elif label == prev_label:
                    prefix = 'I-'; mark = 'Y'
                else:
                    prefix = 'B-'; mark = 'Y'
                if token == '"':
                    token = '""""'
                # print(f'{token}\t{prefix}{label}\t{mark}\tSP: {sp}\tSentence: {sentence_id}\tDoc: {doc.id}')
                print(f'{token}\t{prefix}{label}\t{mark}\tCol: {col.id}\tDoc: {doc.id}')
                sp += 1
                prev_label = label
            sentence_id += 1
            print()

# print(tokeniser.tokenise_spans(text_cm))
session.logout()
