import os
import json
import pandas as pd

def compare_folders(dev_folder, test_folder, prod_folder):
    comparison_results = {}

    for file_name in ['a.json', 'b.json', 'c.json']:
        dev_file = os.path.join(dev_folder, file_name)
        test_file = os.path.join(test_folder, file_name)
        prod_file = os.path.join(prod_folder, file_name)

        comparison_result = compare_json_files(dev_file, test_file, prod_file)
        comparison_results[file_name] = comparison_result

    # Save the comparison results to an Excel file
    save_comparison_results(comparison_results)

def compare_json_files(file1, file2, file3):
    # Load JSON data from files
    with open(file1, 'r') as file:
        data1 = json.load(file)

    with open(file2, 'r') as file:
        data2 = json.load(file)

    with open(file3, 'r') as file:
        data3 = json.load(file)

    # Compare JSON data
    comparison_result1 = compare_dicts(data1, data2, 'dev', 'test')
    comparison_result2 = compare_dicts(data2, data3, 'test', 'prod')

    return comparison_result1 + comparison_result2

def compare_dicts(dict1, dict2, label1, label2, current_key=''):
    result = []

    for key in set(dict1.keys()).union(dict2.keys()):
        new_key = f"{current_key}.{key}" if current_key else key

        if key in dict1 and key in dict2:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                result.extend(compare_dicts(dict1[key], dict2[key], label1, label2, current_key=new_key))
            elif dict1[key] != dict2[key]:
                result.append((new_key, dict1[key], f'{label1}.{key}', dict2[key], f'{label2}.{key}'))
        elif key in dict1:
            result.append((new_key, dict1[key], f'{label1}.{key}', None, None))
        else:
            result.append((new_key, None, None, dict2[key], f'{label2}.{key}'))

    return result

def save_comparison_results(comparison_results):
    for file_name, result in comparison_results.items():
        label1, label2 = result[0][2], result[0][4]  # Extract labels from the first result
        df = pd.DataFrame(result, columns=['Key', f'Value in {label1}', f'Value in {label2}'])
        output_file = f'{file_name}_comparison_result.xls'
        df.to_excel(output_file, index=False)
        print(f"Comparison result for {file_name} saved to {output_file}")

# Example usage
dev_folder = 'path/to/dev'
test_folder = 'path/to/test'
prod_folder = 'path/to/prod'
compare_folders(dev_folder, test_folder, prod_folder)
