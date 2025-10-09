from helper_functions.feature_extraction import TraceUtils
from helper_functions.feature_extraction.deviancecommon import xes_to_positional, xes_to_data_positional

def abstract_split(log, train_id, test_id, log_conversion=None, id_extractor=None):
    """
    Abstract function performing splitting

    :param log:                 loaded XES log
    :param train_id:            Id of the rows belonging to the training set
    :param test_id:             Id of the rows belonging to the testing set
    :param log_conversion:      Whether we need to convert the log to a specific representation. Otherwise, the log is preserved as it is
    :param id_extractor:        The function used to extract the id for a given trace in the transformed log
    :return:
    """
    if id_extractor is None:
        id_extractor = TraceUtils.getTraceId
    converted_log = log
    if log_conversion is not None:
        converted_log = log_conversion(log)

    assert isinstance(train_id, set)
    assert isinstance(test_id, set)
    train_log = []
    test_log = []

    for trace in converted_log:
        traceId = id_extractor(trace)
        if traceId in train_id:
            train_log.append(trace)
        elif traceId in test_id:
            test_log.append(trace)
        else:
            assert False

    assert len(train_log) > 0
    assert len(test_log) > 0
    return train_log, test_log

def xes_to_data_propositional_split(log, ids=None, doForce=False):
    """
    Convert log to data-aware propositional representation. If `ids` is None,
    process the full log.
    """
    if ids is None:
        return xes_to_data_positional(log, forceSomeElements=doForce, label=True)
    else:
        # Legacy compatibility - single output to match current usage
        return [xes_to_data_positional(log, forceSomeElements=doForce, label=True)], None

def xes_to_propositional_split(log, ids=None):
    """
    Convert log to propositional representation. If `ids` is None,
    process the full log.
    """
    if ids is None:
        return xes_to_positional(log, label=True)
    else:
        # Legacy compatibility
        return [xes_to_positional(log, label=True)], None

def xes_to_tracelist_split(log, ids=None):
    """
    Return canonical trace list. If `ids` is None, return entire log.
    """
    if ids is None:
        return log
    else:
        return [log], None
