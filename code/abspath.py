import os


def AbsolutePath(local_filename, data_folder = "data"):
    """
    AbsolutePath gets the absolute path of the file given the name of the folder containing the data
    and the name of the file inside that folder and assuming that the repository contains a data folder
    and a code folder.

    Arguments:
    local_filename (str): name of the data file inside the data folder
    data_folder (str): name of the folder which contains the data, default "data"

    Result:
    str: the function returns the absolute path of the selected file
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))     # path of the code
    
    # Construct the absolute path to the data directory relative to the code directory
    data_dir = os.path.join(script_dir, "..", data_folder)
    
    # Construct the absolute path to the data file
    data_file_path = os.path.join(data_dir, local_filename)
    
    return data_file_path
