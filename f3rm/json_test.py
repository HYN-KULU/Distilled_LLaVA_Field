import json

# Your JSON string
json_string = '''
{
    "target_object": "wooden block",
    "reference_object": "baymax"
}
'''

# Parse the JSON string into a Python dictionary
data = json.loads(json_string)

# Now you can access the values using keys
target_object = data['target_object']
reference_object = data['reference_object']

print("Target object:", target_object)
print("Reference object:", reference_object)
