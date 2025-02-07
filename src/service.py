from typing import Dict, List
from collections import defaultdict

from src.data import DataManager
from src.evaluation.metrics import Metrics
from src.evaluation.metrics_display import MetricsDisplay
from src.clients import Qdrant
from src.config import App
from loguru import logger

from src.models.model_manager import ModelManager


class Service:
    def __init__(self):
        self.config = App.config()
        self.qdrant = Qdrant(self, self.config)
        self.model_manager = ModelManager(self, self.config)
        self.data_manager = DataManager(self, self.config)
        self.groups: Dict[str, Dict[str, Dict[str, List[int]]]] = defaultdict(lambda: defaultdict(dict))
        self.metrics: Dict[str, dict] = defaultdict(lambda: defaultdict(dict))

    def _get_next_group_number(self, key, language):
        return len(self.groups[key][language]) + 1

    def _run(self):
        try:
            logger.info("Initializing Service..")
            self.init()

            logger.info("Loading Data..")
            self.data_manager.load_data()

            for language, df in self.data_manager.data.items():
                for key, value in self.model_manager.models.items():
                    check = self.qdrant.check_collection(f"{key}-{language}")
                    if check:
                        self.qdrant.delete_collection(f"{key}-{language}")

            logger.debug("Begin Processing Data..")
            for language, df in self.data_manager.data.items():
                for key, value in self.model_manager.models.items():
                    if "/" in key and len(key.split("/")) > 1:
                        key = key.split("/")[1]

                    self.qdrant.create_collection(f"{key}-{language}", value.dim)

                    self._process_initial_pass(df, value, f"{key}-{language}", language, key)

                    self._process_final_pass(value, f"{key}-{language}", language, key)

                    self._finalize_groups()

                    self._create_metric(key, language)

            self._pretty_print_metrics()

            self._show_overall_metrics()



        except Exception as e:
            logger.error(e)
            raise e

    def _pretty_print_metrics(self):
        logger.debug(f"{len(self.metrics)}, {self.metrics}")

        for model, languages in self.metrics.items():
            for language, metrics in languages.items():
                logger.info(f"Metrics for {model} in {language}:")
                for key, value in metrics.items():
                    logger.info(f"{key}: {value}")


    def _show_overall_metrics(self):
        metrics_display = MetricsDisplay(self.metrics)
        metrics_display.display_metrics()

    def _create_metric(self, model, language):
        try:
            groups = []
            validation = []

            metric = Metrics(language, model)

            for _, row in self.data_manager.validation_data.iterrows():
                validation.append([int(id) for id in row['IDs'].replace("'", "").split(",")])

            for group_ids in self.groups[model][language].values():
                groups.append(group_ids)

            if model not in self.metrics:
                self.metrics[model] = {}
            self.metrics[model][language] = metric.create_metrics(groups, validation)

            logger.debug(f"Metrics: {self.metrics}")
        except Exception as e:
            logger.error(e)
            raise e

    def _process_initial_pass(self, df, model, collection_name, language, key):
        try:
            for _, row in df.iterrows():
                logger.debug("Starting to insert data..")
                current_id = int(row['ID'])

                # Create Point Structure
                payload = {
                    "id": current_id,
                    "subject": row['SUBJECT'],
                    "message": row['MESSAGE'],
                }

                # Use the Subject and Message as a vector string
                vector_string = f"{row['SUBJECT']}, {row['MESSAGE']}"

                # Create an embedding
                embedding = self.model_manager.create_embedding(model.name, vector_string)

                # Check for similarity
                similarity = self.qdrant.query_points(collection_name, embedding)

                # Perform Grouping
                if similarity:
                    logger.info(
                        f"Duplicate found for {current_id} with similarities: {', '.join([f'id: {point.id}, score: {point.score}' for point in similarity])}"
                    )
                    self._handle_similarity(current_id, similarity, key, language)
                else:
                    logger.debug(f"No similarity found for {current_id}")
                    self._create_new_group(current_id, key, language)

                    self.qdrant.insert_point(
                        collection_name,
                        self.qdrant.create_point(current_id, embedding, payload)
                    )

        except Exception as e:
            logger.error(e)
            raise e

    def _create_new_group(self, current_id, key, language):
        """
        Create a new group for the current id

        Event: No similarity found for the current id

        :param current_id:
        :param key:
        :param language:
        :return:
        """
        try:
            new_group_name = f"Group{self._get_next_group_number(key, language)}"
            self.groups[key][language][new_group_name] = [current_id]
            logger.debug(f"Created new group {new_group_name} for {current_id}")
        except Exception as e:
            logger.error(e)
            raise

    def _handle_similarity(self, current_id, similarity, key, language):
        """
        Handle similarity for the current id, when similarity is found. Group the current id with the similar ids.

        Event: Similarity found for the current id

        :param current_id:
        :param similarity:
        :param key:
        :param language:
        :return:
        """
        try:
            target_groups = self._find_target_groups(similarity, key, language)

            if target_groups:
                primary_group = target_groups[0]
                self.groups[key][language][primary_group].append(current_id)

                if len(target_groups) > 1:
                    self._merge_groups(target_groups, key, language)
                logger.debug(f"Added {current_id} to group {primary_group} under {key}/{language}")
            else:
                new_group_name = f"Group{self._get_next_group_number(key, language)}"
                similar_ids = [int(p.id) if isinstance(p.id, str) else p.id for p in similarity]
                self.groups[key][language][new_group_name] = [current_id] + similar_ids
                logger.debug(f"Created new group {new_group_name} under {key}/{language}")
        except Exception as e:
            logger.error(e)
            raise e

    def _find_target_groups(self, similarity, key, language):
        """

        Find the target groups for the current id based on similarity with other ids.

        Event: Similarity found for the current id

        :param similarity:
        :param key:
        :param language:
        :return:
        """

        try:
            target_groups = []
            for point in similarity:
                point_id = int(point.id) if isinstance(point.id, str) else point.id
                if key in self.groups and language in self.groups[key]:
                    for group_name, group_ids in self.groups[key][language].items():
                        if point_id in group_ids and group_name not in target_groups:
                            target_groups.append(group_name)
            return target_groups
        except Exception as e:
            logger.error(e)
            raise e


    def _merge_groups(self, target_groups, key, language):

        """

        Merge the target groups into a single group for the current id based on similarity with other ids.

        Event: Similarity found for the current id

        :param target_groups:
        :param key:
        :param language:
        :return:
        """
        try:
            primary_group = target_groups[0]
            for group_name in target_groups[1:]:
                self.groups[key][language][primary_group].extend(
                    self.groups[key][language][group_name]
                )
                del self.groups[key][language][group_name]
            logger.debug(f"Merged {len(target_groups)} groups into {primary_group}")
        except Exception as e:
            logger.error(e)
            raise e



    def _create_temp_group(self, current_id, key, language):
        """
        Create a temporary group for the current id

        Event: Similarity found for the current id

        :param current_id:
        :param key:
        :param language:
        :return:
        """
        try:
            temp_group_name = f"TempGroup{self._get_next_group_number(key, language)}"
            self.groups[key][language][temp_group_name] = [current_id]
            logger.debug(f"Created temporary group {temp_group_name} for {current_id}")
        except Exception as e:
            logger.error(e)
            raise e



    def _process_final_pass(self, model, collection_name, language, key):
        """
        Process the final pass for the current id

        Event: No similarity found for the current id

        :param model:
        :param collection_name:
        :param language:
        :param key:
        :return:
        """

        try:
            singletons = [group_id for group_ids in self.groups[key][language].values() if len(group_ids) == 1 for
                          group_id
                          in group_ids]
            for sid in singletons:
                vector = self.qdrant.get_point(collection_name, sid)
                similarity = self.qdrant.query_points(collection_name, vector.vector, score_threshold=0.8)
                if similarity:
                    self._handle_similarity(sid, similarity, key, language)
        except Exception as e:
            logger.error(e)
            raise e



    def cleanup(self):
        """
        Clean up the service
        :return:
        """
        try:
            logger.info("Cleaning up..")
            for language, df in self.data_manager.data.items():
                for key, value in self.model_manager.models.items():
                    self.qdrant.delete_collection(f"{key}-{language}")
        except Exception as e:
            logger.error(e)

    def run(self):
        try:
            self._run()
            self._finalize_groups()
        except Exception as e:
            logger.error(f"Error running service: {e}")
            self.cleanup()
            raise e

    def _finalize_groups(self):
        for key in self.groups:
            for language in self.groups[key]:
                final_groups = {}
                for i, (_, group_ids) in enumerate(sorted(self.groups[key][language].items()), 1):
                    final_groups[f"Group{i}"] = sorted(set(group_ids))
                self.groups[key][language] = final_groups

    def init(self):
        try:
            self.qdrant.health_check()
            self.model_manager.setup()
            logger.success("Setups for Models successfully completed.")
        except Exception as e:
            logger.error(f"Error setting up models: {e}")
            raise e
