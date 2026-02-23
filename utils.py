import json
import matplotlib.pylab as plt
from pathlib import Path
from mne.io import read_raw_ctf
from mne.channels import read_layout
from mpl_toolkits.axes_grid1 import make_axes_locatable



def colorbar(mappable, label):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, label=label)
    plt.sca(last_axes)
    return cbar



def check_many(multiple, target, func=None):
    """
    Check for the presence of strings in a target string.

    Parameters
    ----------
    multiple : list
        List of strings to be found in the target string.
    target : str
        The target string in which to search for the specified strings.
    func : str
        Specifies the search mode: "all" to check if all strings are present, or "any" to check if
        any string is present.

    Notes
    -----
    - This function works well with `if` statements in list comprehensions.
    """

    func_dict = {
        "all": all, "any": any
    }
    if func in func_dict:
        use_func = func_dict[func]
    else:
        raise ValueError("pick function 'all' or 'any'")
    check_ = []
    for i in multiple:
        check_.append(i in target)
    return use_func(check_)


def get_files(target_path, suffix, strings=(""), prefix=None, check="all", depth="all"):
    """
    Return a list of files with a specific extension, prefix, and name containing specific strings.

    Searches either all files in the target directory or within a specified directory.

    Parameters
    ----------
    target_path : str or pathlib.Path or os.Path
        The most shallow searched directory.
    suffix : str
        File extension in "*.ext" format.
    strings : list of str
        List of strings to search for in the file name.
    prefix : str
        Limits the output list to file names starting with this prefix.
    check : str
        Specifies the search mode: "all" to check if all strings are present, or "any" to check if
        any string is present.
    depth : str
        Specifies the depth of the search: "all" for recursive search, "one" for shallow search.

    Returns
    -------
    subdirs : list
        List of pathlib.Path objects representing the found files.
    """

    path = Path(target_path)
    files = []
    if depth == "all":
        files = [file for file in path.rglob(suffix)
                 if file.is_file() and file.suffix == suffix[1:] and
                 check_many(strings, file.name, check)]
    elif depth == "one":
        files = [file for file in path.iterdir()
                 if file.is_file() and file.suffix == suffix[1:] and
                 check_many(strings, file.name, check)]

    if isinstance(prefix, str):
        files = [file for file in files if file.name.startswith(prefix)]
    files.sort(key=lambda x: x.name)
    return files


def get_directories(target_path, strings=(""), check="all", depth="all"):
    """
    Return a list of directories in the path (or all subdirectories) containing specified strings.

    Parameters
    ----------
    target_path : str or pathlib.Path or os.Path
        The most shallow searched directory.
    depth : str
        Specifies the depth of the search: "all" for recursive search, "one" for shallow search.

    Returns
    -------
    subdirs : list
        List of pathlib.Path objects representing the found directories.
    """

    path = Path(target_path)
    subdirs = []
    if depth == "all":
        subdirs = [subdir for subdir in path.glob("**/")
                   if subdir.is_dir() and check_many(strings, str(subdir), check)]
    elif depth == "one":
        subdirs = [subdir for subdir in path.iterdir()
                   if subdir.is_dir() and check_many(strings, str(subdir), check)]
    # pylint: disable=unnecessary-lambda
    subdirs.sort(key=lambda x: str(x))
    return subdirs


def make_directory(root_path, extended_dir):
    """
    Create a directory along with intermediate directories.

    Parameters
    ----------
    root_path : str or pathlib.Path or os.Path
        The root directory.
    extended_dir : str or list
        Directory or directories to create within `root_path`.

    Returns
    -------
    root_path : str or pathlib.Path or os.Path
        The updated root directory.
    """

    root_path = Path(root_path)
    if isinstance(extended_dir, list):
        root_path = root_path.joinpath(*extended_dir)
    else:
        root_path = root_path.joinpath(extended_dir)

    root_path.mkdir(parents=True, exist_ok=True)
    return root_path


def save_dict_as_json(file_path, dictionary):
    """Saves a dictionary as a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(dictionary, json_file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving dictionary: {e}")


def update_json_file(file_path, update_dict):
    """Updates an existing JSON file with a dictionary. Replaces values of existing keys."""
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        
        data.update(update_dict)
        
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error updating dictionary: {e}")


def load_json(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
    return data


def find_missing_channels(raw, layout="CTF275", ch_name_string="M"):
    """Returns missing channels and the indices"""
    lay_full = read_layout(fname=layout)
    all_chan = lay_full.names
    raw_chan = [i for i in raw.info["ch_names"] if i[0] == ch_name_string]
    missing_chan = list(set(all_chan) - set(raw_chan))
    missing_chan_ix = lay_full.pick(picks=missing_chan).ids

    return missing_chan, missing_chan_ix
