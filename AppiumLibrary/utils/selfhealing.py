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

    def __init__(self, host='localhost', port=27017, update_healed_locator=False, similarity_percentage=0.93,
                 healing_strategy='xpath', heal_with_llm=False):
        self.update_healed_locator = update_healed_locator
        self.similarity_percentage = similarity_percentage
        self.healing_strategy = healing_strategy
        self.heal_with_llm = heal_with_llm
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
        # except Exception as e:
        #     logging.error("An unexpected error occurreddddd:", e)

    def check_if_locator_with_identifier(self, locator):
        ef = ElementFinder()
        identifiers = [a_key + '=' for a_key in ef._strategies.keys()]
        for identifier in identifiers:
            if str(locator).startswith(identifier):
                return True
        return False

    def add_locator_to_database(self, elements, locator, locator_variable_name, current_activity, old_locator=None,
                                app_package=None):
        """Adds a found element and its associated metadata to the database.
            @param elements: Element Value
            @param locator: Locator Value
            @param locator_variable_name: Variable Name of Locator
            @param current_activity: Current App Activity running on
            @param old_locator: The Old locator value before self-healing to be updated
        """
        self.app_package = app_package
        self.collection = self.database[self.app_package]
        existing_name = None
        existing_locator = None
        try:

            item = {
                "name": locator_variable_name,
                "locator": locator,
                "activity": current_activity,
                "tag": elements.get_attribute("classname"),
                "attributes": {
                    "text": elements.text,
                    "package": elements.get_attribute("package"),
                    "resource-id": elements.get_attribute("resource-id"),
                    "bounds": f"{elements.location}, {elements.size}",
                    # "bounds": f"{elements.get_attribute('bounds')}",
                    "content-desc": elements.get_attribute("content-desc"),
                },
                "created-at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last-time-passed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            if locator_variable_name:
                existing_name = self.collection.find_one(
                    {"name": locator_variable_name})
                print(f"existingName is: {existing_name}")
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
                logging.info(
                    "Locator with same variable name exists in Database, Updating locator value...")
                self.collection.update_one({"name": locator_variable_name},
                                           {"$set": {"locator": locator,
                                                     "last-time-passed": datetime.now().strftime(
                                                         "%Y-%m-%d %H:%M:%S")}})
            if existing_locator:
                logging.info(
                    "Locator with same locator criteria exists in Database, Updating locator value...")
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
                                        " Please set update_healed_locator=${True} for auto update xxxx"
                                        f"Locator is : {locator} and old locator is: {old_locator}")
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
                                        " Please set update_healed_locator=${True} for auto update")
                logging.info(f"Document inserted successfully with ID: {result.inserted_id}")

            if old_locator and old_locator != locator:
                if self.update_healed_locator:
                    logging.warning("Updating locators in Robot Files")
                    self.update_healed_locator_variable(old_locator_value=str(old_locator),
                                                        new_locator_value=str(locator))
                else:
                    logging.warning("Update healed locator switched off")
        except errors.ConnectionFailure as e:
            logging.error(f"MongoDB not connected due to error: {e}")
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while connecting to MongoDB due to error xxx : {e}")

    def select_locator_from_database(self, locator_variable_name, locator, app_package):
        self.app_package = app_package
        self.collection = self.database[self.app_package]
        print(f"self.collection is: {self.collection}")
        try:
            print(f"Locator Var Name are Name: {locator_variable_name} and locator: {locator}")
            existing_name = self.collection.find_one(
                {"name": locator_variable_name}) if locator_variable_name else None
            existing_locator = self.collection.find_one({"locator": locator})
            print(f"Search Criteria are Name: {existing_name} and locator: {existing_locator}")
            if existing_name:
                return {'tag': existing_name['tag'], 'attributes': existing_name['attributes']}
            else:
                return self.handle_locator_with_no_variables()
            if existing_locator:
                return {'tag': existing_locator['tag'], 'attributes': existing_locator['attributes']}

                # logging.warning("Self-Healing Algorithm Couldn't find any related elements")
        except errors.ConnectionFailure as e:
            logging.error(f"MongoDB not connected due to error: {e}")
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while connecting to MongoDB due to error : {e}")

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

    def restructure_json(self, data, current_activity, keys_to_keep=None):
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
                        "activity": current_activity,
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

    def find_most_similar(self, obj1, obj_list):
        """
        Find the most similar object from a list based on the given object and calculate its similarity percentage.

        Args:
            obj1 (dict): The object to compare against the list.
            obj_list (list): A list of objects to compare with.

        Returns:
            dict, float: The most similar object and its similarity percentage.
        """

        def calculate_similarity(source_element_object, target_element_object):
            # Weight distribution
            weights = {
                "tag": 20,  # 20% for tag similarity
                "attributes": {
                    "text": 15,  # 15% for text attribute
                    "resource-id": 25,  # 25% for resource ID attribute
                    "bounds": 20,  # 20% for bounds attribute
                    "content-desc": 20  # 20% for content description attribute
                }
            }

            # Check for package mismatch
            source_pkg = source_element_object.get("attributes", {}).get("package",
                                                                         source_element_object.get(
                                                                             "attributes", {})
                                                                         .get("package", ""))
            target_pkg = target_element_object.get("attributes", {}).get("package",
                                                                         target_element_object.get(
                                                                             "attributes", {})
                                                                         .get("package", ""))
            if source_pkg != target_pkg:
                return 0.0  # No similarity if package mismatch

            # Initialize similarity score
            similarity = 0

            # Compare tags
            if source_element_object["tag"] == target_element_object["tag"]:
                similarity += weights["tag"]  # Exact match
            else:
                similarity += 0  # Completely different tags

            # Compare attributes
            source_element_attributes = source_element_object.get(
                "attributes", {})
            target_element_attributes = target_element_object.get(
                "attributes", {})
            for key, weight in weights["attributes"].items():
                source_value = source_element_attributes.get(key, "")
                target_value = target_element_attributes.get(key, "")
                if key == "bounds":
                    # Consider bounds equivalence
                    similarity += weight if str(
                        source_value) == str(target_value) else 0
                else:
                    similarity += weight if source_value == target_value else 0

            return similarity

        # Find the most similar object
        most_similar_object = None
        highest_similarity = 0.0

        for obj2 in obj_list:
            similarity = calculate_similarity(obj1, obj2)
            if similarity > highest_similarity:
                most_similar_object = obj2
                highest_similarity = similarity

        return most_similar_object, highest_similarity

    def apply_self_healing_by_llm(self, application, locator):
        logging.info("Attempting Healing Locator Using LLM...")
        elements_in_page = self.restructure_json(json.loads(self.xml_to_json(application.page_source)),
                                                 application.current_activity)
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant. \
                                                your task is to analyze the provided data and return the most similar"
                                              f" locator to the failed locator. \
                                                You have to return only the locator as a string using"
                                              f" {self.healing_strategy}, do not return any other text "
                                              f"or code or block."},
                {
                    "role": "user",
                    "content": f"Failed Locator is: {locator} and Elements in Page are: {elements_in_page}"
                }
            ]
        )
        logging.info(completion.choices[0].message.content)
        return completion.choices[0].message.content

    def apply_self_healing(self, application, variable_name, locator, window_size, app_package):
        print("Attempting Self Healing ....")
        if self.heal_with_llm:
            return self.apply_self_healing_by_llm(application, locator)
        else:
            return self.apply_self_healing_similarity(application, variable_name, locator, window_size, app_package)

    def apply_self_healing_similarity(self, application, variable_name, locator, window_size, app_package):
        logging.info("Attempting Healing Locator...")
        elements_in_page = self.restructure_json(json.loads(self.xml_to_json(application.page_source)),
                                                 application.current_activity)
        healing_candidate = self.select_locator_from_database(
            variable_name, locator, app_package)
        logging.info(f"Healing Candidate From Database: {healing_candidate}")
        if isinstance(healing_candidate, list):
            for a_healing_candidate in healing_candidate:
                if a_healing_candidate:
                    healed_locator = self.find_closest_locator(
                        a_healing_candidate, elements_in_page, window_size)
                    if healed_locator:
                        print(f"Attributes in healed locators are: {healed_locator['attributes'].values()}")
                        return self.locator_reconstruction(application, self.healing_strategy, healed_locator)
                    logging.info("Could not apply self healing, No Element matching found")
        else:
            if healing_candidate:
                healed_locator = self.find_closest_locator(
                    healing_candidate, elements_in_page, window_size)
                if healed_locator:
                    print(f"Attributes in healed locators are: {healed_locator['attributes'].values()}")
                    return self.locator_reconstruction(application, self.healing_strategy, healed_locator)
            logging.info("Could not apply self healing, No Element matching found")
        return None

    def update_healed_locator_variable(self, old_locator_value, new_locator_value, file_extensions=['.robot']):
        updated_files = []
        current_directory = os.getcwd()
        for root, _, files in os.walk(current_directory):

            for file_name in files:

                if any(file_name.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file_name)

                    with open(file_path, "r") as file:
                        content = file.read()
                    if old_locator_value in content:
                        updated_content = re.sub(
                            re.escape(old_locator_value), new_locator_value, content)
                        with open(file_path, "w") as file:
                            file.write(updated_content)

                        updated_files.append(file_path)
                        print(f"Healed Locators updated in: {file_path} successfully")

        return updated_files

    # New Similarity Method

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
        print(f"Source Bounds: {source_bounds}")
        print(f"Target Bounds: {target_bounds}")
        print(f"Bounds Similarity is: {1 - sum(differences) / len(differences)} and text is: {element_text}")
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
        print(f"source_attributes['text'] is: {source_attributes['text']}")
        print(f"target_attributes['text'] is: {target_attributes['text']}")
        print(f"Source attributes are: {source_attributes.keys()}")
        print(f"Target attributes are: {target_attributes.keys()}")
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
        print(f"Attr similarity is: {total_similarity / count if count > 0 else 0}")
        if bounds_source_attribute: source_attributes.update({'bounds': bounds_source_attribute})
        if bounds_target_attribute: target_attributes.update({'bounds': bounds_target_attribute})

        return total_similarity / count if count > 0 else 0

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

            if 'bounds' in candidate['attributes'].keys():
                # handle bounds in iOS
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
                    print("Got Element by Bounds Similarity")
                    best_match = candidate
        print(f"all_bounds_similarity: {bounds_all_similarity} and max is {max(bounds_all_similarity)} "
              f"and max attr is: {max(all_similarity_att)} and best match is: {best_match} "
              f"and overall_similarity is : {all_over_all_similarity} and max is {max(all_over_all_similarity)}")
        highest_similar_element_attribute = self.filter_locator_attributes(candidate_locators
                                                                           [all_similarity_att.
                                                                           index(max(all_similarity_att)) + 1]
                                                                           ['attributes'])

        if not best_match:
            logging.warning(
                f"Self-Healing Algorithm couldn't find element similar to : {target_locator['attributes']} "
                f"with similarity percentage:"
                f" {self.similarity_percentage}. However, the highest similar element attributes are: "
                f"{highest_similar_element_attribute}, "
                f"decrease similarity percentage to: {round(max(all_similarity_att), 2)} "
                f"for applying self-Healing to this element")
        return best_match

    def filter_locator_attributes(self, attributes):
        filtered_attributes = {}
        for key in ['package', 'text', 'resource-id', 'class']:
            if key in attributes and attributes[key].strip():
                filtered_attributes[key] = attributes[key]
        return filtered_attributes

    def convert_bounds_str_to_tuple(self, bounds_str):
        """
        Convert bounds string in format '[x1,y1][x2,y2]' to a tuple (x1, y1, x2, y2).

        :param bounds_str: String representing bounds, e.g., '[944,1186][1370,1423]'.
        :return: Tuple of integers, e.g., (944, 1186, 1370, 1423).
        """
        # Split the string into the two parts and parse them as dictionaries
        if "{" in bounds_str:
            parts = bounds_str.split("},")

            # Fix each part to form valid dictionary strings
            dict1 = ast.literal_eval(parts[0] + "}")  # Add closing curly brace
            # Add opening curly brace
            dict2 = ast.literal_eval(parts[1].strip())

            # Extract values
            x, y = dict1['x'], dict1['y']
            # TODO: IOS Handling for x,y,width and height
            width, height = dict2['width'] + x, dict2['height'] + y

            return (x, y, width, height)
        parts = bounds_str.replace('[', '').replace(']', ',').split(',')
        parts = [part for part in parts if part != '']
        # Convert to integers and return as a tuple
        return tuple(map(int, parts))

    def locator_reconstruction(self, application, strategy, element):
        logging.info(f"Reconstructing Locator with strategy: {strategy} and element: {element}")
        platform = application.desired_capabilities['platformName'].lower()
        logging.info(f"Platform is: {platform}")

        if strategy == 'accessibility_id':
            strategy_key = f'accessibility_id_{platform}'
        else:
            strategy_key = strategy

        # Return locator based on strategy and available attributes
        if strategy_key == 'accessibility_id_ios' and 'name' in element['attributes']:
            logging.info(f"Returning Locator with Accessibility ID Strategy: {element['attributes']['name']}")
            return f"identifier={element['attributes']['name']}"
        elif strategy_key == 'id' and 'resource-id' in element['attributes']:
            logging.info(f"Returning Locator with ID Strategy: {element['attributes']['resource-id']}")
            return f"id={element['attributes']['resource-id']}"
        elif strategy_key == 'accessibility_id_android' and 'content-desc' in element['attributes']:
            logging.info(f"Returning Locator with Accessibility ID Strategy: {element['attributes']['content-desc']}")
            return f"accessibility_id={element['attributes']['  content-desc']}"
        elif strategy_key == 'name' and 'name' in element['attributes']:
            logging.info(f"Returning Locator with Name Strategy: {element['attributes']['name']}")
            return f"name={element['attributes']['name']}"

        # Fallback to xpath if strategy not supported or attributes missing
        if platform == 'ios':
            logging.info(f"Returning Locator with Xpath Strategy: {element['attributes']['bounds']}")
            return f"//{element['tag']}[@x='{element['attributes']['x']}'][@y='{element['attributes']['y']}'][@width='{element['attributes']['width']}'][@height='{element['attributes']['height']}']"
        else:  # android
            logging.info(f"Returning Locator with Xpath Strategy: {element['attributes']['bounds']}")
            return f"//{element['tag']}[@bounds='{element['attributes']['bounds']}']"
