from pymongo import MongoClient, errors
from datetime import datetime
import logging
import json
import xml.etree.ElementTree as eT


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

    def __init__(self, host='localhost', port=27017, update_healed_locator=False):
        self.database = None
        self.collection = None
        self.update_healed_locator = update_healed_locator
        try:
            self.client = MongoClient(
                host=host,
                port=port,
            )
            logging.info("Connected to MongoDB!")
            self.database = self.client["ROBOT_ELEMENTS"]
            self.collection = self.database["elements"]

        except errors.ServerSelectionTimeoutError as e:
            logging.error("MongoDB connection timed out:", e)
        except errors.ConnectionFailure as e:
            logging.error("Failed to connect to MongoDB:", e)
        except errors.OperationFailure as e:
            logging.error("Operation failed on MongoDB:", e)
        except Exception as e:
            logging.error("An unexpected error occurred:", e)

    def add_locator_to_database(self, elements, locator, locator_variable_name, current_activity, old_locator=None):
        """Adds a found element and its associated metadata to the database.
            @param elements: Element Value
            @param locator: Locator Value
            @param locator_variable_name: Variable Name of Locator
            @param current_activity: Current App Activity running on
        """
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
                    "content-desc": elements.get_attribute("content-desc"),
                },
                "created-at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last-time-passed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            if locator_variable_name:
                existing_name = self.collection.find_one({"name": locator_variable_name})
            else:
                existing_locator = self.collection.find_one({"locator": locator})

            if existing_name:
                logging.info("Locator with same variable name exists in Database, Updating locator value...")
                self.collection.update_one({"name": locator_variable_name},
                                           {"$set": {"locator": locator,
                                                     "last-time-passed": datetime.now().strftime(
                                                         "%Y-%m-%d %H:%M:%S")}})
                if self.update_healed_locator:
                    if old_locator and old_locator!=locator:
                        self.update_healed_locator_variable(old_locator_value=str(old_locator),
                                                            new_locator_value=str(locator))
                else:
                    if old_locator and old_locator != locator:
                        logging.warning("Locators are changed but not updated in Files."
                                     " Please set update_healed_locator='enabled' for auto update")
            else:
                if existing_locator:
                    logging.warning(f"Locator is found with no variable name. Please add variable name to {locator}")
                    return
                result = self.collection.insert_one(item)
                if self.update_healed_locator:
                    if old_locator and old_locator != locator:
                        self.update_healed_locator_variable(old_locator_value=str(old_locator),
                                                            new_locator_value=str(locator))
                else:
                    if old_locator and old_locator != locator:
                        logging.warning("Locators are changed but not updated in Files."
                                        " Please set update_healed_locator='enabled' for auto update")
                logging.info(f"Document inserted successfully with ID: {result.inserted_id}")

        except errors.ConnectionFailure as e:
            logging.error(f"Database not connected: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurredddd: {e}")

    def select_locator_from_database(self, locator_variable_name, locator):
        existing_name = self.collection.find_one({"name": locator_variable_name})
        existing_locator = self.collection.find_one({"locator": locator})
        if existing_name:
            return {'tag': existing_name['tag'], 'attributes': existing_name['attributes']}
        else:
            if existing_locator:
                return {'tag': existing_locator['tag'], 'attributes': existing_locator['attributes']}
            logging.warning("Couldn't find any related elements")
            return None

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
                    filtered_attributes = {k: v for k, v in attributes.items() if k in keys_to_keep}
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
                                                                         source_element_object.get("attributes", {})
                                                                         .get("package", ""))
            target_pkg = target_element_object.get("attributes", {}).get("package",
                                                                         target_element_object.get("attributes", {})
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
            source_element_attributes = source_element_object.get("attributes", {})
            target_element_attributes = target_element_object.get("attributes", {})
            for key, weight in weights["attributes"].items():
                source_value = source_element_attributes.get(key, "")
                target_value = target_element_attributes.get(key, "")
                if key == "bounds":
                    # Consider bounds equivalence
                    similarity += weight if str(source_value) == str(target_value) else 0
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

    def apply_self_healing(self, application, variable_name, locator):
        logging.info("Attempting Healing Locator...")
        elements_in_page = self.restructure_json(json.loads(self.xml_to_json(application.page_source)),
                                                 application.current_activity,
                                                 keys_to_keep=["package", "text", "resource-id", "bounds",
                                                               "content-desc"])
        healing_candidate = self.select_locator_from_database(variable_name, locator)
        logging.info(f"Healing Candidate From Database: {healing_candidate}")
        if healing_candidate:
            healing_candidate, similarity_score = self.find_most_similar(healing_candidate, elements_in_page)
            logging.info(f"Similarity Score between failed element and most similar element is: {similarity_score}")
            if similarity_score >= 50:
                # TODO: new xpath how it will be constructed???
                locator = f"//{healing_candidate['tag']}[@resource-id='{healing_candidate['attributes']['resource-id']}']"
                return locator
        logging.info("Could not apply self healing")
        return None

    def update_healed_locator_variable(self, old_locator_value, new_locator_value, file_extension=['.robot']):
        updated_files = []
        current_directory = os.getcwd()
        for root, _, files in os.walk(current_directory):

            for file_name in files:

                if any(file_name.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file_name)

                    with open(file_path, "r") as file:
                        content = file.read()
                    if old_locator_value in content:

                        updated_content = re.sub(re.escape(old_locator_value), new_locator_value, content)
                        with open(file_path, "w") as file:
                            file.write(updated_content)

                        updated_files.append(file_path)
                        print(f"Healed Locators updated in: {file_path} successfully")

        return updated_files
