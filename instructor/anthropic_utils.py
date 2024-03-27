# type: ignore
import re
from typing import Any, Dict, Type, TypeVar

from pydantic import BaseModel

try:
    import xml.etree.ElementTree as ET

    import xmltodict
    import warnings

    warnings.warn(
        "xmltodict and xml.etree.ElementTree modules not found. Please install them to proceed. `pip install xmltodict`",
        ImportWarning,
        stacklevel=2,
    )


T = TypeVar("T", bound=BaseModel)


def json_to_xml(model: Type[BaseModel]) -> str:
    """Takes a Pydantic model and returns XML format for Anthropic function calling."""
    model_dict = model.model_json_schema()

    root = ET.Element("tool_description")
    tool_name = ET.SubElement(root, "tool_name")
    tool_name.text = model_dict.get("title", "Unknown")
    description = ET.SubElement(root, "description")
    description.text = "This is the function that must be used to construct the response."
    parameters = ET.SubElement(root, "parameters")
    references = model_dict.get("$defs", {})
    list_type_found = _add_params(parameters, model_dict, references)

    if list_type_found:  # Need to append to system prompt for List type handling
        return (
            ET.tostring(root, encoding="unicode")
            + "\nFor any List[] types, include multiple <$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME> tags for each item in the list. XML tags should only contain the name of the parameter."
        )
    else:
        return ET.tostring(root, encoding="unicode")


def _add_params(
    root: ET.Element, model_dict: Dict[str, Any], references: Dict[str, Any]
) -> bool:  # Return value indiciates if we ever came across a param with type List
    # TODO: handling of nested params with the same name
    properties = model_dict.get("properties", {})
    list_found = False
    nested_list_found = False

    for field_name, details in properties.items():
        parameter = ET.SubElement(root, "parameter")
        name = ET.SubElement(parameter, "name")
        name.text = field_name
        type_element = ET.SubElement(parameter, "type")

        # Get type
        if "anyOf" in details:  # Case where there can be multiple types
            # supports:
            # case 1: List type (example json: {'anyOf': [{'items': {'$ref': '#/$defs/PartialUser'}, 'type': 'array'}, {'type': 'null'}], 'default': None, 'title': 'Users'})
            # case 2: nested model (example json: {'anyOf': [{'$ref': '#/$defs/PartialDate'}, {'type': 'null'}], 'default': {}})
            field_type = " or ".join(
                [d["type"] if "type" in d else (d["$ref"] if "$ref" in d else "unknown") for d in details["anyOf"]]
            )
        else:
            field_type = details.get(
                "type", "unknown"
            )  # Might be better to fail here if there is no type since pydantic models require types

        if "array" in field_type and "items" not in details:
            raise ValueError("Invalid array item.")

        # Check for nested List
        if "array" in field_type and "$ref" in details["items"]:
            type_element.text = f"List[{details['title']}]"
            list_found = True
            nested_list_found = True
        # Check for non-nested List
        elif "array" in field_type and "type" in details["items"]:
            type_element.text = f"List[{details['items']['type']}]"
            list_found = True
        else:
            type_element.text = field_type

        param_description = ET.SubElement(parameter, "description")
        param_description.text = details.get("description", "")

        if isinstance(details, dict) and "$ref" in details:  # Checking if there are nested params
            reference = _resolve_reference(references, details["$ref"])

            if "enum" in reference:
                type_element.text = reference["type"]
                enum_values = reference["enum"]
                values = ET.SubElement(parameter, "values")
                for value in enum_values:
                    value_element = ET.SubElement(values, "value")
                    value_element.text = value
                continue

            nested_params = ET.SubElement(parameter, "parameters")
            list_found |= _add_params(
                nested_params,
                reference,
                references,
            )
        elif field_type == "array" and nested_list_found:  # Handling for List[] type
            nested_params = ET.SubElement(parameter, "parameters")
            list_found |= _add_params(
                nested_params,
                _resolve_reference(references, details["items"]["$ref"]),
                references,
            )

    return list_found


def _resolve_reference(references: Dict[str, Any], reference: str) -> Dict[str, Any]:
    parts = reference.split("/")[2:]  # Remove "#" and "$defs"
    for part in parts:
        references = references[part]
    return references


def extract_xml(content: str) -> str:  # Currently assumes 1 function call only
    """Extracts XML content in Anthropic's schema from a string."""
    pattern = r"<function_calls>.*?</function_calls>"
    matches = re.findall(pattern, content, re.DOTALL)
    return "".join(matches)


def xml_to_model(model: Type[T], xml_string: str) -> T:
    """Converts XML in Anthropic's schema to an instance of the provided class."""
    parsed_xml = xmltodict.parse(xml_string)
    model_dict = parsed_xml["function_calls"]["invoke"]["parameters"]
    return model(**model_dict)  # This sometimes fails if Anthropic's response hallucinates from the schema


def build_xml_from_schema(
    schema: Dict[str, Any], parent: ET | None = None, root_name: str = "tool_description", defs: Dict[str, Any] | None = None
) -> str:
    if parent is None:
        root = ET.Element(root_name)
        tool_name = ET.SubElement(root, "tool_name")
        tool_name.text = schema.get("title", "")
        if "description" in schema:
            description = ET.SubElement(root, "description")
            description.text = schema["description"]
    else:
        root = parent

    if defs is None:
        defs = schema.get("$defs", {})

    if "properties" in schema:
        properties = ET.SubElement(root, "properties")
        for key, value in schema["properties"].items():
            property_elem = ET.SubElement(properties, "property")
            name = ET.SubElement(property_elem, "name")
            name.text = key
            type_elem = ET.SubElement(property_elem, "type")
            type_elem.text = value.get("type", "object")

            # Recursively handle nested objects
            if "$ref" in value:
                ref = _resolve_reference(defs, value["$ref"])
                if "enum" in ref:
                    type_elem.text = ref["type"]
                    values = ET.SubElement(property_elem, "values")
                    for enum_value in ref["enum"]:
                        value_elem = ET.SubElement(values, "value")
                        value_elem.text = enum_value
            elif "enum" in value:
                type_elem.text = value["type"]
                values = ET.SubElement(property_elem, "values")
                for enum_value in value["enum"]:
                    value_elem = ET.SubElement(values, "value")
                    value_elem.text = enum_value
            elif value.get("type") == "object":
                build_xml_from_schema(value, property_elem, defs=defs)
            elif value.get("type") == "array":
                type_elem.text = "List"
                if "items" in value:
                    items = value["items"]
                    if items.get("type") == "object":
                        # Handle object items in the list
                        item_properties = ET.SubElement(property_elem, "items")
                        build_xml_from_schema(items, item_properties, defs=defs)
                    elif items.get("type") == "array":
                        # Handle nested arrays
                        type_elem.text += f"[{items.get('title', '')}]"
                        if "items" in items:
                            item_properties = ET.SubElement(property_elem, "items")
                            build_xml_from_schema(items["items"], item_properties, defs=defs)
                    elif "$ref" in items:
                        # Handle references to nested models in the list
                        ref_path = items["$ref"].split("/")
                        ref_name = ref_path[-1]
                        if ref_name in defs:
                            item_properties = ET.SubElement(property_elem, "items")
                            build_xml_from_schema(defs[ref_name], item_properties, defs=defs)
                    else:
                        # Handle simple types in the list
                        type_elem.text += f"[{items.get('type', '')}]"
            # Handle references to nested models
            elif "$ref" in value:
                ref_path = value["$ref"].split("/")
                ref_name = ref_path[-1]
                if ref_name in defs:
                    build_xml_from_schema(defs[ref_name], property_elem, defs=defs)
            # Handle allOf and anyOf within properties
            elif "allOf" in value or "anyOf" in value:
                for construct in ["allOf", "anyOf"]:
                    if construct in value:
                        for sub_schema in value[construct]:
                            if "$ref" in sub_schema:
                                ref_path = sub_schema["$ref"].split("/")
                                ref_name = ref_path[-1]
                                if ref_name in defs:
                                    build_xml_from_schema(defs[ref_name], property_elem, defs=defs)
                            else:
                                build_xml_from_schema(sub_schema, property_elem, defs=defs)
            else:
                raise ValueError(f"Unsupported schema")

    return ET.tostring(root, encoding="utf-8").decode("utf-8")
