from pymongo import MongoClient, errors
from datetime import datetime
import logging
import json
import xml.etree.ElementTree as ET

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
    def __init__(self, host='localhost', port=27017, username='admin', password='adminpassword', auth_Source="admin"):
        try:
            # Create a MongoDB client
            self.client = MongoClient(
                host=host,
                port=port,
                username=username,
                password=password,
                authSource=auth_Source
            )
            logging.info("Connected to MongoDB!")

        except errors.ServerSelectionTimeoutError as e:
            logging.error("MongoDB connection timed out:", e)
        except errors.ConnectionFailure as e:
            logging.error("Failed to connect to MongoDB:", e)
        except errors.OperationFailure as e:
            logging.error("Operation failed on MongoDB:", e)
        except Exception as e:
            logging.error("An unexpected error occurred:", e)

    def add_locator_to_database(self, elements, locator, locator_variable_name):
        """Adds a found element and its associated metadata to the database.

        Parameters
        ----------
        elements : list
            List Of Elements Returned
        locator : str
            locator => {id, xpath, etc...}
        locator_variable_name : str
            the varibale name in robot file
        """
        db = self.client["ROBOT_ELEMENTS"]
        collection = db["elements"]
        item = {
            "name": locator_variable_name,
            "locator": locator,
            "tag": elements[0].get_attribute("classname"),
            "attributes": {
            "text": elements[0].text,
            "package": elements[0].get_attribute("package"),
            "resource-id": elements[0].get_attribute("resource-id"),
            "bounds": f"{elements[0].location}, {elements[0].size}",
            "content-desc": elements[0].get_attribute("content-desc")
            },
            "created-at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last-time-passed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        existing_document = collection.find_one({"name": locator_variable_name, 
                                                     "locator": locator})
        if existing_document:
            print("Document already exists in the collection.")
        else:
            result = collection.insert_one(item)
            logging.info(f"Document inserted successfully with ID: {result.inserted_id}")
            
    def select_locator_from_database(self, locator_variable_name):
        db = self.client["ROBOT_ELEMENTS"]
        collection = db["elements"]
        existing_document = collection.find_one({"name": locator_variable_name})
        if existing_document:
            return {'tag': existing_document['tag'], 'attributes':existing_document['attributes']}
        else:
            logging.info("Couldn't find any related elements")
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
        root = ET.fromstring(xml_string)
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
                    filtered_attributes = {k: v for k, v in attributes.items() if k in keys_to_keep}
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
    
    
    def find_most_similar(self, obj1, obj_list):
        """
        Find the most similar object from a list based on the given object and calculate its similarity percentage.

        Args:
            obj1 (dict): The object to compare against the list.
            obj_list (list): A list of objects to compare with.

        Returns:
            dict, float: The most similar object and its similarity percentage.
        """
        def calculate_similarity(obj1, obj2):
            # Weight distribution
            weights = {
                "tag": 20,  # 20% for tag similarity
                "attributes": {
                    "text": 15,          # 15% for text attribute
                    "resource-id": 25,   # 25% for resource ID attribute
                    "bounds": 20,        # 20% for bounds attribute
                    "content-desc": 20   # 20% for content description attribute
                }
            }

            # Check for package mismatch
            package1 = obj1.get("attributes", {}).get("package", obj1.get("attributes", {}).get("package", ""))
            package2 = obj2.get("attributes", {}).get("package", obj2.get("attributes", {}).get("package", ""))
            if package1 != package2:
                return 0.0  # No similarity if package mismatch

            # Initialize similarity score
            similarity = 0

            # Compare tags
            if obj1["tag"] == obj2["tag"]:
                similarity += weights["tag"]  # Exact match
            else:
                similarity += 0  # Completely different tags

            # Compare attributes
            attributes1 = obj1.get("attributes", {})
            attributes2 = obj2.get("attributes", {})
            for key, weight in weights["attributes"].items():
                value1 = attributes1.get(key, "")
                value2 = attributes2.get(key, "")
                if key == "bounds":
                    # Consider bounds equivalence
                    similarity += weight if str(value1) == str(value2) else 0
                else:
                    similarity += weight if value1 == value2 else 0

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