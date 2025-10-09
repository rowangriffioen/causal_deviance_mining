"""
Python wrapper to extract sequence encodings from logs.
Requires GoSwift.jar in the same folder as it runs it.
Might need to change VMOptions dependent on the version of Java the machine is running on

"""
import subprocess
from helper_functions.feature_extraction.PathUtils import *
import os


JAR_NAME = "GoSwift.jar"  # Jar file to run
#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#JAR_PATH   = os.path.join(SCRIPT_DIR, "GoSwift.jar")


#OUTPUT_FOLDER = "outputlogs/"  # Where to put output files
#INPUT_FOLDER = "logs/"  # Where input logs are located

## This is needed for one java version.. One for Java 8 and other for later
#VMoptions = " --add-modules java.xml.bind"
VMOptions = ""
## All parameters to run the program with.


def create_output_filename(input_log, name):
    """
    Create output json file name corresponding to the trial parameters
    :param input_log: input log filenae
    :param name: name of the trial
    :return:
    """
    prefix = input_log
    if input_log.endswith(".xes"):
        prefix = prefix[:prefix.find(".xes")]
    return f"{prefix}_{name}.json"


def create_call_params(inp_path, results_folder, paramString, inputFile=None, outputFile=None):
    params = paramString.split()

    if outputFile:
        params.append("--outputFile")
        #outputPath = os.path.join(results_folder, "outputlogs")
        os.makedirs(results_folder, exist_ok=True)
        params.append(os.path.join(results_folder, outputFile))
    if inputFile:
        params.append("--logFile")
        params.append(os.path.join(inp_path, inputFile[0]))
        if inputFile[1]:
            params.append("--requiresLabelling")

    return params

def create_call_params2(inp_path, results_folder, paramString, inputFile=None, outputFile=None):
    params = paramString.split()
    #if outputFile:
    params.append("--outputFile")
    #outputPath = os.path.join(results_folder, "outputlogs")
    os.makedirs(results_folder, exist_ok=True)
    params.append(os.path.join(results_folder, outputFile))
    #if inputFile:
    params.append("--logFile")
    # point at the actual XES in inp_path:
    params.append(os.path.join(inp_path, inputFile))
    #    if inputFile[1]:
    #        params.append("--requiresLabelling")
    return params

def call_params(inp_path, results_folder, paramString, inputFile, outputFile, err_logger):
    """
    Function to call java subprocess
    TODO: Send sigkill when host process (this one dies) to also kill the subprocess calls
    :param paramString:
    :param inputFile:
    :return:
    """

    print("Started working on {}".format(inputFile[0]))
    parameters = create_call_params(inp_path, results_folder, paramString, inputFile, outputFile)
    FNULL = open(os.devnull, 'w')  # To write output to devnull, we dont care about it

    # os.chdir(SCRIPT_DIR)
    # os.makedirs("output", exist_ok=True)

    # No java 8
    #subprocess.call(["java", "-jar", "--add-modules", "java.xml.bind", JAR_NAME] + parameters, stdout=FNULL,
    #                stderr=open("errorlogs/error_" + outputFile, "w"))  # blocking

    print(" ".join(["java", "-jar",  JAR_NAME] + parameters))
    # Java 8
    subprocess.call(["java", "-jar",  JAR_NAME] + parameters, stdout=FNULL,
                    stderr=open(os.path.join(err_logger, "error_" + outputFile), "w"))  # blocking

    print("Done with {}".format(str(parameters)))


# def move_files(split_nr, folder, results_folder):
#     """
#     Move generated encodings
#     :param split_nr: number of cv split
#     :param folder: folder for encoding at end location
#     :param results_folder: resulting folder
#     :return:
#     """
#     # source = './output/'
#     # dest1 = './' + results_folder + '/split' + str(split_nr) + "/" + folder + "/"
#     #
#     # files = os.listdir(source)
#     #
#     # ## Moves all files in the folder to detination
#     # for f in files:
#     #     shutil.move(source+f, dest1)

def genParamStrings(sequence_threshold):
    return [
        ("--coverageThreshold {} ".format(sequence_threshold) + "--featureType Sequence --minimumSupport 0.1 --patternType MR --encodingType Frequency", "SequenceMR", "mr"),
        ("--coverageThreshold {} ".format(sequence_threshold) + "--featureType Sequence --minimumSupport 0.1 --patternType MRA --encodingType Frequency", "SequenceMRA", "mra"),
        ("--coverageThreshold {} ".format(sequence_threshold) + "--featureType Sequence --minimumSupport 0.1 --patternType TR --encodingType Frequency", "SequenceTR", "tr"),
        ("--coverageThreshold {} ".format(sequence_threshold) + "--featureType Sequence --minimumSupport 0.1 --patternType TRA --encodingType Frequency", "SequenceTRA", "tra"),
    ]

# def run_sequences(inp_path, log_path, results_folder, err_logger, max_splits, sequence_threshold=5):
    # """
    # Runs GoSwift.jar with 4 different parameter sets to create sequential encodings.
    # Expects split files named <base>_<i>.xes under inp_path, matching cross_validation outputs.
    # Returns a list of strategy folder names.
    # """
    # paramStrings = genParamStrings(sequence_threshold)
    # strategies = []

    # for paramString, techName, folder in paramStrings:
    #     print(f"Working on {techName} @{folder}")
    #     strategies.append(folder)

    #     for splitNr in range(max_splits):
    #         # Prepare output subfolder
    #         outputPath = feature_extraction.FileNameUtils.embedding_path(splitNr, results_folder, folder)
    #         os.makedirs(outputPath, exist_ok=True)

    #         # Derive the correct split filename: base_{splitNr+1}.xes
    #         base, ext = os.path.splitext(log_path)
    #         split_filename = f"{base}_{splitNr+1}{ext}"
    #         inputFile = (split_filename, False)

    #         # Build GoSwift output filename and call parameters
    #         outputFilename = create_output_filename(inputFile[0], techName)
    #         call_params(inp_path, outputPath, paramString, inputFile, outputFilename, err_logger)

    #         # Optional: move leftover files from './output/' if PathUtils.move_files is available
    #         try:
    #             move_files('./output/', results_folder, splitNr + 1, folder)
    #         except NameError:
    #             pass

    # return strategies


def run_sequences(inp_path, log_path, results_folder, err_logger, sequence_threshold=5):
    """
    Runs GoSwift.jar with each parameter set to create sequential encodings
    for the single event log at `log_path`.
    Returns a list of strategy folder names.
    """
    # 1) generate the four (paramString, techName, folder) tuples
    param_strings = genParamStrings(sequence_threshold)
    strategies = []

    # 2) loop over each encoding “strategy”
    for param_string, tech_name, folder in param_strings:
        print(f"Working on {tech_name} → {folder}")
        strategies.append(folder)

        # 3) make one output subfolder per strategy
        output_path = os.path.join(results_folder, folder)
        os.makedirs(output_path, exist_ok=True)

        # 4) point to your single input file by basing off log_path
        input_file = (os.path.basename(log_path), False)

        # 5) build the GoSwift output filename, then invoke it
        output_filename = create_output_filename(input_file[0], tech_name)
        call_params(inp_path, output_path, param_string, input_file, output_filename, err_logger)

        # 6) if you still need to collect any stray files from './output/'
        try:
            move_files('./output/', results_folder, None, folder)
        except NameError:
            # no move_files defined, so skip
            pass

    return strategies


def generateSequences(
    inp_path,
    log_path,
    results_folder,
    sequence_threshold=5,
    err_logger=None
):
    """
    Runs GoSwift.jar on a single XES log (no splits), captures errors,
    and moves the produced globalLog.csv into place.
    """
    import subprocess
    import os

    # Prepare error-logging directory (if requested)
    if err_logger:
        os.makedirs(err_logger, exist_ok=True)

    # Remember where we started
    orig_cwd = os.getcwd()
    # Jump into the folder where this script (and GoSwift.jar) live
    script_dir = os.path.dirname(__file__)
    os.chdir(script_dir)

    # Ensure GoSwift’s own ./output folder is clean
    mkdir_test('./output/')

    yamlPart = {}
    for paramString, techName, folder in genParamStrings(sequence_threshold):
        outputFilename = create_output_filename(log_path, techName)
        params = create_call_params2(inp_path, results_folder, paramString, log_path, outputFilename)

        print(f"→ Running {techName}:")
        print("   " + " ".join(["java", "-jar", JAR_NAME] + params))

        # Redirect stdout to devnull, stderr into our err_logger
        stderr_stream = None
        if err_logger:
            stderr_path = os.path.join(err_logger, f"error_{techName}.log")
            stderr_stream = open(stderr_path, "w")

        retcode = subprocess.call(
            ["java", "-jar", JAR_NAME] + params,
            stdout=subprocess.DEVNULL,
            stderr=stderr_stream
        )
        if stderr_stream:
            stderr_stream.close()

        if retcode != 0:
            print(f"⚠️  Java returned code {retcode} for {techName} (see {stderr_path})")

        # Move the GoSwift ./output → your results_folder/<folder>
        # use the absolute path to the module’s ./output folder:
        output_dir = os.path.join(script_dir, 'output')
        res = move_files(output_dir, results_folder, 1, folder)
        yamlPart[folder] = os.path.abspath(os.path.join(res, "globalLog.csv"))

    # Go back to wherever we were
    os.chdir(orig_cwd)
    return yamlPart