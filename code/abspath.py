import os


def abs_path(local_filename, data_folder):
    """
    abs_path gets the absolute path of the file given the name of the folder containing the data
    and the name of the file inside that folder and assuming that the repository contains a data folder
    and a code folder.

    :param local_filename: name of the data file
    :type local_filename: str
    :param data_folder: name of the folder which contains the data
    :type data_folder: str
    :return: the function returns the absolute path of the selected file
    :rtype: str
    
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))     # path of the code
    
    # Construct the absolute path to the data directory relative to the code directory
    data_dir = os.path.join(script_dir, "..", data_folder)
    
    # Construct the absolute path to the data file
    data_file_path = os.path.join(data_dir, local_filename)
    
    return data_file_path
