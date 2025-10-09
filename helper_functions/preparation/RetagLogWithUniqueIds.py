import os

from opyenxes.data_in.XUniversalParser import XUniversalParser
from opyenxes.factory.XFactory import XFactory
from opyenxes.data_out.XesXmlSerializer import XesXmlSerializer

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

def generateStringFromNumber(n):
    defaults = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnoprstuvwxyz0123456789"
    return "".join(map(lambda x: defaults[x], numberToBase(n, len(defaults))))

def changeLog(logFileName, output_dir=None, doQualityCheck=True, i=0):
    """
    Change the dataset by ensuring unique IDs to each trace and write to the specified directory.

    :param logFileName: path to the raw XES log file
    :param output_dir: directory where the unique log file will be written (defaults to current working directory)
    :param doQualityCheck: whether to perform a uniqueness check on trace IDs
    :param i: starting index for ID generation
    :return: tuple(final_file_path, log_object)
    """
    # Determine output directory
    if output_dir is None:
        output_dir = os.getcwd()
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Construct final file path
    base_name = os.path.splitext(os.path.basename(logFileName))[0]
    final_file_name = os.path.join(output_dir, f"{base_name}_unique.xes")

    # If already exists, just load and return
    if os.path.isfile(final_file_name):
        with open(final_file_name, 'r') as file:
            log = XUniversalParser().parse(file)[0]
        return final_file_name, log

    # Parse and retag
    with open(logFileName, 'r') as file:
        log = XUniversalParser().parse(file)[0]
        for trace in log:
            trace.get_attributes()["concept:name"] = XFactory.create_attribute_literal(
                "concept:name", generateStringFromNumber(i)
            )
            for event in trace:
                if "lifecycle:transition" not in event.get_attributes():
                    event.get_attributes()["lifecycle:transition"] = XFactory.create_attribute_literal(
                        "lifecycle:transition", "COMPLETE"
                    )
            i += 1

    # Serialize to new file
    with open(final_file_name, 'w') as file2:
        XesXmlSerializer().serialize(log, file2)

    # Optional quality check
    if doQualityCheck:
        with open(final_file_name, 'r') as file2:
            log2 = XUniversalParser().parse(file2)[0]
            ids = [t.get_attributes()["concept:name"].get_value() for t in log2]
            assert len(ids) == len(set(ids)), "Duplicate trace IDs found!"

    return final_file_name, log
