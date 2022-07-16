import torch
import torchtext

def text_tokenize(s):
    """Tokenize input sentence
    
    Convert a sentence from csv file to reversed list of words

    Args:
        s (string): a sentence from csv file

    Returns: 
        list of string: list of words converted from a sentence
    """
    split = s.strip().split(' ')
    split.reverse()
    return split


def form_tokenize(s):
    """Tokenize output formula

    Convert a formula from csv file to list of words

    Args:
        s (string): a formula from csv file

    Returns: 
        list of string: list of words converted from a sentence
    """
    return s.strip().split(' ')


def generate_field():
    """Generate torchtext.data.Field for input and output

    Generate torchtext.data.Field

    Args:
        None

    Returns:
        tuple of torchtext.data.Field: Field for input setntence and output formula
    """
    text_field = torchtext.data.Field(
        sequential=True,
        use_vocab=True,
        init_token='<S>',
        eos_token='<E>',
        fix_length=None,
        dtype=torch.long,
        preprocessing=None,
        postprocessing=None,
        lower=True,  
        tokenize=text_tokenize,
        include_lengths=True,
        pad_first=False,
        truncate_first=False,
        stop_words=None,
        is_target=False
    )
    form_field = torchtext.data.Field(
        sequential=True,
        use_vocab=True,
        init_token='<S>',
        eos_token='<E>',
        fix_length=None,
        dtype=torch.long,
        preprocessing=None,
        postprocessing=None,
        lower=True,  
        tokenize=form_tokenize,
        include_lengths=True,
        pad_first=False,
        truncate_first=False,
        stop_words=None,
        is_target=True
    )
    return text_field, form_field


def generate_dataset(path, format, fields, skip_header):
    """Generate dataset

    Generate dataset from a file indicated by path

    Args:
        path (string): A path to the data file
        format (string): the data file format
        fields (list of (string, torchtext.data.Field)): text_field and form_field
        skip_header (bool): whether skipping the first row of the data file

    Returns:
        torchtext.data.TabularDataset: dataset
    """
    return torchtext.data.TabularDataset(
        path=path,
        format=format,
        fields=fields,
        skip_header=skip_header
    )


def generate_iterator(dataset, batch_size, train, sort_key=None):
    """Generate iterator

    Generate iterator from dataset 

    Args:
        dataset (torchtext.data.TabularDataset): dataset
        batch_size (int): batch size
        train (bool): whether for training or not
        sortkey (function): a key to use sorting examples in order to make batch properly

    Returns:
        torchtext.data.Iterator: iterator
    """
    return torchtext.data.Iterator(
        dataset, batch_size, train=train, sort=False, sort_key=sort_key
    )