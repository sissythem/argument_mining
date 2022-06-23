import copy
from collections import defaultdict
from string import punctuation


class TrackedAnnotation:
    def __init__(self, tokens):
        self.tokens = tokens
        self.matches = []
        self.match_history = []
        self.completed = False

    def __repr__(self):
        return str(self.tokens)

    def __str__(self):
        return str(self.tokens)

    def match_token(self, token):
        # always check beginning as well
        if 0 not in self.matches:
            self.matches += [0]

        matched = False
        for i, idx in enumerate(self.matches):
            if len(self.tokens) <= idx:
                continue
            if self.tokens[idx] == token or (
                    (isinstance(token, int) or token.isdigit()) and str(token) == str(self.tokens[idx])):
                self.matches[i] = idx + 1
                self.print_current_match(idx + 1)
                matched = True
                if idx + 1 == len(self.tokens):
                    self.completed = True

        return matched

    def print_current_match(self, match_up_to):
        return
        print("MATCH:|||", end=" ")
        for t, tok in enumerate(self.tokens):
            if t + 1 == match_up_to:
                print("[" + tok + "]", end=" ")
            elif t < match_up_to:
                print("(" + tok + ")", end=" ")
            else:
                print(tok, end=" ")
        print()

    def get_tracking_error(self):
        return len(self.tokens) - self.get_best_match()

    def get_best_match(self):
        return max(self.matches)


class AnnotationTracker:
    """Class to match a current annotated sequence to a set of ground truth annotations"""

    def __init__(self, candidates, min_tracking_error):
        self.annotations = [TrackedAnnotation(x) for x in candidates]
        self.min_tracking_error = min_tracking_error

    def finish_tracking_sequence(self):
        self.indices = [0 for _ in self.annotations]

    def finish_tracking_annotation(self):
        # update best effort matches
        for i in range(len(self.indices)):
            if not self.tracked[i] and self.indices[i] > self.best_effort_match[i]:
                self.best_effort_match[i] = self.indices[i]
        # reset
        self.indices = [0 for _ in self.annotations]

    def check_equality(self, cand, idx, token):
        if cand[idx] == token:
            return True
        # if it's the last token, and there only punct. differences at the end, match it anyway
        if idx == len(cand) - 1 and cand[idx].rstrip(punctuation) == token.rstrip(punctuation):
            return True
        return False

    def advance(self, token):
        matched_something = False
        for i, cand in enumerate(self.annotations):
            if cand.match_token(token):
                matched_something = True

        if not matched_something:
            for i, cand in enumerate(self.annotations):
                matched_something = matched_something or cand.match_token(token)
            raise ValueError(f"Could not track token: {token}")

    def check_sanity(self):
        # check all annotations have been tracked
        tracking_errors = []
        for i, ann in enumerate(self.annotations):
            if not ann.completed:
                err = ann.get_tracking_error()
                best = ann.get_best_match()
                print(f"Sanity check: incomplete match with error={err}: (", best, " / ", len(ann.tokens), ")")
                print([f"[{tok}]" if t < best else tok for (t, tok) in enumerate(ann.tokens)])
                tracking_errors.append(err)

        assert all(te <= self.min_tracking_error for te in tracking_errors)
