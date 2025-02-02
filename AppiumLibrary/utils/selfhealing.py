from pymongo import MongoClient, errors
from datetime import datetime
import logging
import json
import xml.etree.ElementTree as eT
import ast
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
from AppiumLibrary.locators.elementfinder import ElementFinder
import re
import os
from openai import OpenAI


class SelfHealing:
    """ A utility class for self-healing locators in Robot framework AppiumLibrary.

    The SelfHealing class is designed to enhance the robustness of automated test cases
    by dynamically identifying and updating locators that fail during execution. When
    a locator cannot be found, the class attempts to self-heal by leveraging alternative
    strategies, such as:
    - Searching for similar elements based on attributes.
    - Analyzing the DOM for recently updated or altered locators.

    This class helps reduce test flakiness due to minor changes in the application's UI
    or DOM structure.
    """

    def __init__(self, host='localhost', port=27017, update_healed_locator=False, update_file_extensions=None,
                 similarity_percentage=0.93, healing_strategy='xpath', heal_with_llm=False, OpenAI_key=None,
                 llm_model='gpt-4o-mini'):
        self.update_healed_locator = update_healed_locator
        self.update_file_extensions = update_file_extensions
        self.similarity_percentage = similarity_percentage
        self.healing_strategy = healing_strategy
        self.heal_with_llm = heal_with_llm
        self.platform = None
        self.open_ai_key = OpenAI_key
        self.llm_model = llm_model
        try:
            self.client = MongoClient(
                host=host,
                port=port,
                # username="admin",
                # password="adminpassword"
            )
            logging.info("Connected to MongoDB!")
            self.database = self.client["ROBOT_ELEMENTS"]
            self.collection = None
            self.app_package = None

        except errors.ServerSelectionTimeoutError as e:
            logging.error(f"MongoDB connection timed out: {e}")
        except errors.ConnectionFailure as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
        except errors.OperationFailure as e:
            logging.error(f"Operation failed on MongoDB: {e}")

    def set_platform(self, platform):
        self.platform = platform
        logging.info(f"Platform is set with value: {self.platform}")

    def check_if_locator_with_identifier(self, locator):
        ef = ElementFinder()
        identifiers = [a_key + '=' for a_key in ef._strategies.keys()]
        for identifier in identifiers:
            if str(locator).startswith(identifier):
                return True
        return False

    def add_locator_to_database(self, elements, locator, locator_variable_name, old_locator=None, app_package=None):
        """Adds a found element and its associated metadata to the database.
            @param elements: Element Value
            @param locator: Locator Value
            @param locator_variable_name: Variable Name of Locator
            @param old_locator: The Old locator value before self-healing to be updated
        """
        self.app_package = app_package
        self.collection = self.database[self.app_package]
        existing_name = None
        existing_locator = None

        try:
            if self.platform == 'ios':
                item = {
                    "name": locator_variable_name,
                    "locator": locator,
                    "tag": elements.get_attribute("type"),
                    "attributes": {
                        "text": elements.get_attribute('label'),
                        "package": self.app_package,
                        "name": elements.get_attribute("name"),
                        "bounds": f"{elements.location}, {elements.size}",
                        # "bounds": f"{elements.get_attribute('bounds')}"
                    },
                    "created-at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "last-time-passed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            elif self.platform == 'android':
                item = {
                    "name": locator_variable_name,
                    "locator": locator,
                    "tag": elements.get_attribute("classname"),
                    "attributes": {
                        "text": elements.text,
                        "package": self.app_package,
                        "resource-id": elements.get_attribute("resource-id"),
                        "bounds": f"{elements.location}, {elements.size}",
                        "content-desc": elements.get_attribute("content-desc"),
                    },
                    "created-at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "last-time-passed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            if locator_variable_name:
                existing_name = self.collection.find_one(
                    {"name": locator_variable_name})
            else:
                is_locator_with_identifier = self.check_if_locator_with_identifier(
                    locator)
                if is_locator_with_identifier:
                    locator_without_identifier = re.sub(
                        "^(.*?)=", "", locator, 1)
                    existing_locator = self.collection.find_one(
                        {"locator": locator_without_identifier})
                else:
                    existing_locator = self.collection.find_one(
                        {"locator": locator})

            if existing_name:
                logging.info("Locator with same variable name exists in Database, Updating locator value...")
                self.collection.update_one({"name": locator_variable_name},
                                           {"$set": {"locator": locator,
                                                     "last-time-passed": datetime.now().strftime(
                                                         "%Y-%m-%d %H:%M:%S")}})
            if existing_locator:
                logging.info("Locator with same locator criteria exists in Database, Updating locator value...")
                self.collection.update_one({"locator": locator},
                                           {"$set": {
                                               "last-time-passed": datetime.now().strftime(
                                                   "%Y-%m-%d %H:%M:%S")}})

                if self.update_healed_locator:
                    if old_locator and old_locator != locator:
                        self.update_healed_locator_variable(old_locator_value=str(old_locator),
                                                            new_locator_value=str(locator))
                else:
                    if old_locator and old_locator != locator:
                        logging.warning("Locators are changed but not updated in Files."
                                        " Please set update_healed_locator=${True} for auto update. "
                                        f"New Locator is: {locator} and old locator is: {old_locator}")
            else:
                if existing_locator or existing_name:
                    if self.update_healed_locator:
                        if old_locator and old_locator != locator:
                            self.update_healed_locator_variable(old_locator_value=str(old_locator),
                                                                new_locator_value=str(locator))

                    return
                result = self.collection.insert_one(item)
                if old_locator and old_locator != locator:
                    if self.update_healed_locator:
                        self.update_healed_locator_variable(old_locator_value=str(old_locator),
                                                            new_locator_value=str(locator))
                    else:
                        logging.warning("Locators are changed but not updated in Files."
                                        " Please set update_healed_locator=${True} for auto update. "
                                        f"New Locator is: {locator} and old locator is: {old_locator}")
                logging.info(f"Document inserted successfully with ID: {result.inserted_id}")

            if old_locator and old_locator != locator:
                if self.update_healed_locator:
                    logging.warning(f"Updating locators in {self.update_file_extensions} Files")
                    self.update_healed_locator_variable(old_locator_value=str(old_locator),
                                                        new_locator_value=str(locator))
                else:
                    logging.warning("Update healed locator switched off")
        except errors.ConnectionFailure as e:
            logging.error(f"MongoDB not connected due to error: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while connecting to MongoDB due to error: {e}")

    def select_locator_from_database(self, locator_variable_name, locator, app_package):
        self.app_package = app_package
        self.collection = self.database[self.app_package]
        logging.info(f"Current MongoDB Collection is: {self.collection}")
        try:
            existing_name = self.collection.find_one(
                {"name": locator_variable_name}) if locator_variable_name else None
            existing_locator = self.collection.find_one({"locator": locator})
            logging.info(f"Search Criteria are Name: {existing_name} and locator: {existing_locator}")
            if existing_name:
                return {'tag': existing_name['tag'], 'attributes': existing_name['attributes']}
            else:
                return self.handle_locator_with_no_variables()

        except errors.ConnectionFailure as e:
            logging.error(f"MongoDB not connected due to error: {e}")
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while selecting locator from MongoDB : {e}")

    def handle_locator_with_no_variables(self):
        non_var_locators_list = []
        non_variable_locators = self.collection.find({"name": None})
        for non_variable_locator in non_variable_locators:
            non_var_locators_list.append({'tag': non_variable_locator['tag'],
                                          'attributes': non_variable_locator['attributes']})
        return non_var_locators_list

    def parse_element(self, element):
        """Recursively parse XML elements into dictionary objects with nested children."""
        element_dict = {
            "tag": element.tag,
            "attributes": element.attrib
        }
        for child in element:
            if child.tag in element_dict:
                if not isinstance(element_dict[child.tag], list):
                    element_dict[child.tag] = [element_dict[child.tag]]
                element_dict[child.tag].append(self.parse_element(child))
            else:
                element_dict[child.tag] = self.parse_element(child)
        return element_dict

    def xml_to_json(self, xml_string):
        """Convert an XML string into a JSON string."""
        root = eT.fromstring(xml_string)
        parsed_data = self.parse_element(root)
        return json.dumps(parsed_data, indent=4)

    def restructure_json(self, data, keys_to_keep=None):
        """Restructure JSON to ensure the tag and attributes are at the top level."""
        result = []
        if keys_to_keep is None:
            keys_to_keep = []

        def process_element(element):
            if isinstance(element, dict):
                if "tag" in element:
                    attributes = element.get("attributes", {})
                    filtered_attributes = {k: v for k, v in attributes.items()}
                    flat_structure = {
                        "tag": element["tag"],
                        "attributes": filtered_attributes
                    }
                    result.append(flat_structure)
                for key, value in element.items():
                    if isinstance(value, (dict, list)):
                        process_element(value)
            elif isinstance(element, list):
                for item in element:
                    process_element(item)

        process_element(data)
        return result

    def apply_self_healing_by_llm(self, application, locator):
        logging.info("Attempting Healing Locator Using LLM...")
        elements_in_page = self.restructure_json(json.loads(self.xml_to_json(application.page_source)))
        client = OpenAI(api_key=self.open_ai_key)
        completion = client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": f"You are a helpful assistant. \
                                                your task is to analyze the provided data and return the most "
                                              f"similar locator to the failed locator. \
                                                You have to return only the locator "
                                              f"as a string using {self.healing_strategy},"
                                              f" do not return any other text or code or block."},
                {
                    "role": "user",
                    "content": f"Failed Locator is: {locator} and Elements in Page are: {elements_in_page}"
                }
            ]
        )
        logging.info(f"Response from LLM: {completion.choices[0].message.content}")
        return completion.choices[0].message.content

    def apply_self_healing(self, application, variable_name, locator, window_size, app_package):
        logging.info("Attempting Self Healing ....")
        if self.heal_with_llm:
            return self.apply_self_healing_by_llm(application, locator)
        else:
            return self.apply_self_healing_similarity(application, variable_name, locator, window_size, app_package)

    def apply_self_healing_similarity(self, application, variable_name, locator, window_size, app_package):
        logging.info("Attempting Healing Locator...")
        elements_in_page = self.restructure_json(json.loads(self.xml_to_json(application.page_source)))
        healing_candidate = self.select_locator_from_database(
            variable_name, locator, app_package)
        logging.info(f"Healing Candidate From Database is: {healing_candidate}")
        if isinstance(healing_candidate, list):
            for a_healing_candidate in healing_candidate:
                if a_healing_candidate:
                    healed_locator = self.find_closest_locator(
                        a_healing_candidate, elements_in_page, window_size)
                    if healed_locator:
                        return self.locator_reconstruction(self.healing_strategy, healed_locator)
                    logging.info("Could not apply self healing, No Element matching found")
        else:
            if healing_candidate:
                healed_locator = self.find_closest_locator(
                    healing_candidate, elements_in_page, window_size)
                if healed_locator:
                    return self.locator_reconstruction(self.healing_strategy, healed_locator)
            logging.info("Could not apply self healing, No Element matching found")
        return None

    def update_healed_locator_variable(self, old_locator_value, new_locator_value):
        if not self.update_file_extensions:
            self.update_file_extensions = ['.robot']
        updated_files = []
        current_directory = os.getcwd()
        for root, dirs, files in os.walk(current_directory):
            files = [f for f in files if not f[0] == '.']
            dirs[:] = [d for d in dirs if not d[0] == '.']
            for file_name in files:

                if any(file_name.endswith(ext) for ext in self.update_file_extensions):
                    file_path = os.path.join(root, file_name)
                    with open(file_path, "r", encoding="utf-8") as file:
                        if not file_name.endswith('.json'):
                            content = file.read()
                        else:
                            content = json.load(file)
                    if old_locator_value in content:
                        updated_content = re.sub(
                            re.escape(old_locator_value), new_locator_value, content)
                        with open(file_path, "w", encoding="utf-8") as file:
                            file.write(updated_content)

                        updated_files.append(file_path)
                        logging.info(f"Healed Locators updated in: {file_path} successfully")

        return updated_files

    def calculate_relative_bounds(self, bounds: Tuple[int, int, int, int], screen_size: Tuple[int, int]) -> Tuple[
        float, float, float, float]:
        """
        Calculate relative bounds based on screen size.
        :param bounds: A tuple (x, y, width, height).
        :param screen_size: A tuple (screen_width, screen_height).
        :return: Relative bounds as (rel_x, rel_y, rel_width, rel_height).
        """
        x, y, width, height = bounds
        screen_width = screen_size['width']
        screen_height = screen_size['height']
        return x / screen_width, y / screen_height, bounds[2] / screen_width, bounds[3] / screen_height

    def calculate_bounds_similarity(self, source_bounds: Tuple[float, float, float, float],
                                    target_bounds: Tuple[float, float, float, float], element_text) -> float:
        """
        Calculate the similarity between two sets of relative bounds.
        :param source_bounds: Relative bounds (rel_x, rel_y, rel_width, rel_height).
        :param target_bounds: Relative bounds (rel_x, rel_y, rel_width, rel_height).
        :return: Similarity percentage (0 to 1).
        """
        differences = [abs(source - target)
                       for source, target in zip(source_bounds, target_bounds)]

        return 1 - sum(differences) / len(differences)

    def calculate_attribute_similarity(self, source_attributes: Dict[str, str],
                                       target_attributes: Dict[str, str]) -> float:
        """
        Calculate similarity of attributes using SequenceMatcher.
        :param source_attributes: Attributes of the first element.
        :param target_attributes: Attributes of the second element.
        :return: Similarity percentage (0 to 1).
        """
        total_similarity = 0
        count = 0
        bounds_source_attribute = source_attributes['bounds'] if 'bounds' in source_attributes.keys() else None
        bounds_target_attribute = target_attributes['bounds'] if 'bounds' in target_attributes.keys() else None
        if 'bounds' in source_attributes.keys(): del source_attributes['bounds']
        if 'bounds' in target_attributes.keys(): del target_attributes['bounds']

        for key in source_attributes.keys() & target_attributes.keys():
            if source_attributes[key] is not None and target_attributes[key] is not None:
                matcher = SequenceMatcher(
                    None, source_attributes[key], target_attributes[key])
                total_similarity += matcher.ratio()
                count += 1
        if bounds_source_attribute: source_attributes.update({'bounds': bounds_source_attribute})
        if bounds_target_attribute: target_attributes.update({'bounds': bounds_target_attribute})

        return total_similarity / count if count > 0 else 0

    def format_bounds(self, coordinates):
        """
        Convert coordinate tuple to Android bounds format
        Input: ([x1, y1], [x2, y2])
        Output: "[x1,y1][x2,y2]"
        """
        start_point, end_point = coordinates
        return f"[{start_point[0]},{start_point[1]}][{end_point[0]},{end_point[1]}]"

    def find_closest_locator(
            self,
            target_locator: Dict[str, any],
            candidate_locators: List[Dict[str, any]],
            screen_size: Tuple[int, int]
    ) -> Dict[str, any]:
        """
        Find the closest locator based on bounds and attributes similarity.
        :param target_locator: Locator with attributes and bounds (target locator).
        :param candidate_locators: List of locators to compare with.
        :param screen_size: Current device screen size.
        :return: Closest locator or None if no match is above the threshold.
        """
        bounds_tuple = self.convert_bounds_str_to_tuple(
            target_locator['attributes']['bounds'])

        target_bounds = self.calculate_relative_bounds(
            bounds_tuple, screen_size)
        best_match = None
        highest_similarity = 0
        bounds_all_similarity = []
        all_similarity_att = []
        all_over_all_similarity = []
        for candidate in candidate_locators:
            if self.platform == 'ios':
                if 'x' in candidate['attributes'].keys():
                    candidate['attributes']['bounds'] = [int(candidate['attributes']['x']),
                                                         int(candidate['attributes']['y'])], [
                        int(candidate['attributes']['x']) + int(candidate['attributes']['width']),
                        int(candidate['attributes']['y']) + int(candidate['attributes']['height'])]
                    candidate['attributes']['bounds'] = self.format_bounds(candidate['attributes']['bounds'])
                    if 'label' in candidate['attributes'].keys():
                        candidate['attributes']['text'] = candidate['attributes']['label']

            if 'bounds' in candidate['attributes'].keys():
                candidate_bounds_tuple = self.convert_bounds_str_to_tuple(
                    candidate['attributes']['bounds'])
                candidate_bounds = self.calculate_relative_bounds(
                    candidate_bounds_tuple, screen_size)
                bounds_similarity = self.calculate_bounds_similarity(
                    target_bounds, candidate_bounds, candidate['attributes']['text'])
                bounds_all_similarity.append(bounds_similarity)

                attribute_similarity = self.calculate_attribute_similarity(
                    target_locator['attributes'], candidate['attributes']
                )
                all_similarity_att.append(attribute_similarity)

                overall_similarity = (bounds_similarity + attribute_similarity) / 2
                all_over_all_similarity.append(overall_similarity)

                if overall_similarity > highest_similarity and overall_similarity >= self.similarity_percentage:
                    highest_similarity = overall_similarity
                    best_match = candidate
                if bounds_similarity >= 0.99:
                    logging.info("Got Element by Bounds Similarity")
                    best_match = candidate
        logging.info(f"Max of Bounds similarity is {max(bounds_all_similarity)} "
                     f"and Max attribute Similarity is: {max(all_similarity_att)} "
                     f"and Max of Overall Similarity is {max(all_over_all_similarity)}"
                     f"and best match candidate is: {best_match} ")
        highest_similar_element_attribute = self.filter_locator_attributes(candidate_locators
                                                                           [all_similarity_att.
                                                                           index(max(all_similarity_att)) + 1]
                                                                           ['attributes'])

        if not best_match:
            logging.warning(f"Self-Healing Algorithm couldn't find element similar to : "
                            f"{target_locator['attributes']} with similarity percentage:"
                            f" {self.similarity_percentage}. However, the highest similar element attributes are: "
                            f"{highest_similar_element_attribute}, "
                            f"decrease similarity percentage to: {round(max(all_similarity_att), 2)} "
                            f"for applying self-Healing to this element")
        return best_match

    def filter_locator_attributes(self, attributes):
        filtered_attributes = {}
        for key in ['package', 'text', 'resource-id', 'class', 'name']:
            if key in attributes and attributes[key].strip():
                filtered_attributes[key] = attributes[key]
        return filtered_attributes

    def convert_bounds_str_to_tuple(self, bounds_str):
        """
        Convert bounds string in format '[x1,y1][x2,y2]' to a tuple (x1, y1, x2, y2).

        :param bounds_str: String representing bounds, e.g., '[944,1186][1370,1423]'.
        :return: Tuple of integers, e.g., (944, 1186, 1370, 1423).
        """
        if "{" in bounds_str:
            parts = bounds_str.split("},")

            dict1 = ast.literal_eval(parts[0] + "}")
            dict2 = ast.literal_eval(parts[1].strip())

            x, y = dict1['x'], dict1['y']
            # TODO: IOS Handling for x,y,width and height
            width, height = dict2['width'] + x, dict2['height'] + y

            return (x, y, width, height)
        parts = bounds_str.replace('[', '').replace(']', ',').split(',')
        parts = [part for part in parts if part != '']
        return tuple(map(int, parts))

    def locator_reconstruction(self, strategy, element):
        logging.info(f"Reconstructing Locator with strategy: {strategy} and element: {element}")
        logging.info(f"Platform is: {self.platform}")

        if strategy == 'accessibility_id':
            strategy_key = f'accessibility_id_{self.platform}'
        else:
            strategy_key = strategy

        if strategy_key == 'accessibility_id_ios' and 'name' in element['attributes']:
            logging.info(f"Returning Locator with Accessibility ID Strategy: {element['attributes']['name']}")
            return f"identifier={element['attributes']['name']}"
        elif strategy_key == 'id' and 'resource-id' in element['attributes']:
            logging.info(f"Returning Locator with ID Strategy: {element['attributes']['resource-id']}")
            return f"id={element['attributes']['resource-id']}"
        elif strategy_key == 'accessibility_id_android' and 'content-desc' in element['attributes']:
            logging.info(f"Returning Locator with Accessibility ID Strategy: {element['attributes']['content-desc']}")
            return f"accessibility_id={element['attributes']['content-desc']}"
        elif strategy_key == 'name' and 'name' in element['attributes']:
            logging.info(f"Returning Locator with Name Strategy: {element['attributes']['name']}")
            return f"name={element['attributes']['name']}"

        if self.platform == 'ios':
            logging.info(f"Returning Locator with Xpath Strategy: {element['attributes']['bounds']}")
            return f"//{element['tag']}[@x='{element['attributes']['x']}'][@y='{element['attributes']['y']}'][@width='{element['attributes']['width']}'][@height='{element['attributes']['height']}']"
        else:  # android
            logging.info(f"Returning Locator with Xpath Strategy: {element['attributes']['bounds']}")
            return f"//{element['tag']}[@bounds='{element['attributes']['bounds']}']"
