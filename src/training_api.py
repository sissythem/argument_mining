"""
Module for handling model training
"""
from collections import defaultdict
from os.path import join, getsize, exists, basename
import threading
import logging

from src.utils import ObjectLockContext, listfolders, lock_read_file


class ModelLoader:
    def __init__(self, models_base_path, max_concurrent_training=1):
        self.models_base_path = models_base_path
        self.max_concurrent_training = max_concurrent_training

        self.ids_to_paths = defaultdict(dict)
        self.in_progress = defaultdict(list)
        self.lock = threading.Lock()
        self.check_in_progress_lock = threading.Lock()

        self.selected_model = None

    def can_start_training(self):
        current = [x for (k, v) in self.in_progress.items() for x in v]
        return len(current) < self.max_concurrent_training

    def summarize_in_progress(self):
        return ",".join([(k + "-" + str(a)) for (k, v) in self.in_progress.items() for a in v])

    def summarize_available(self):
        return ",".join([(k + "-" + str(a)) for (k, v) in self.ids_to_paths.items() for a in v])

    def is_in_progress(self, model_id: str, model_type: str = None):
        with ObjectLockContext(self.lock):
            value = model_id in self.in_progress
            if value and model_type is not None:
                value = value and model_type in self.in_progress[model_id]
        return value

    def remove_model_in_progress(self, model_id: str, model_type: str):
        del self.in_progress[model_id][model_type]
        if not self.in_progress[model_id]:
            del self.in_progress[model_id]

    def start_training(self, model_id: str, model_type: str):
        with ObjectLockContext(self.lock):
            self.in_progress[model_id].append(model_type)

    def add_model_id(self, id_, mapping):
        with ObjectLockContext(self.lock):
            # remove from in-progress, if there
            if id_ in self.in_progress:
                for model_type in mapping:
                    try:
                        self.in_progress[id_].remove(model_type)
                    except (KeyError, ValueError):
                        pass
                if not self.in_progress[id_]:
                    del self.in_progress[id_]

            logging.debug(f"Adding model to loader: {mapping}")
            if self.selected_model is None:
                self.select_model(id_)
            for k, v in mapping.items():
                self.ids_to_paths[id_][k] = v

    def get_model_config(self, modelid=None):
        return self.ids_to_paths[modelid or self.selected_model]

    def get_model_in_training(self, modelid):
        return self.in_progress[modelid]

    def has_id(self, id_):
        return id_ in self.ids_to_paths

    def select_model(self, id_):
        logging.info(f"Selecting model: [{id_}]")
        self.selected_model = id_

    def load_available_models(self, models_base_path: str):
        for model_id_path in listfolders(models_base_path):
            logging.debug(f"Loading model id from {model_id_path}")
            model_id = basename(model_id_path)
            for mtype_path in listfolders(model_id_path):
                mtype = basename(mtype_path)
                if mtype not in "adu stance rel embedder".split():
                    continue
                self.ids_to_paths[model_id][mtype] = mtype_path

    def check_models_in_progress(self):
        """
        Check if some model finished training
        """
        with ObjectLockContext(self.check_in_progress_lock):
            for model_id, model_types in self.in_progress.items():
                for model_type in model_types:
                    model_path = join(self.models_base_path, model_id, model_type)
                    content = lock_read_file(join(model_path, "status"))
                    if content == "completed":
                        if not self.model_trained_successfully(model_path):
                            logging.error(f"Checking models in progress: training model {model_path} failed.")
                        else:
                            logging.info(
                                f"Checking models in progress: training of model [{model_id}] [{model_type}] complete!")
                            self.add_model_id(model_id, {model_type: model_path})
                        self.remove_model_in_progress(model_id, model_type)

    def model_trained_successfully(self, base_path: str):
        """
        Check if model training concluded successfully
        """
        model_bin = join(base_path, 'pytorch_model.bin')
        return exists(model_bin) and getsize(model_bin) > 0
