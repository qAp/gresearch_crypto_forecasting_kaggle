import datatable


def read_jay(pth):
    return datatable.fread(pth).to_pandas()
