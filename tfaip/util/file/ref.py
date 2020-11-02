def read_utf8_textline_reference_from_file(fn: str) -> str:
    with open(fn, 'r') as ref_file:
        aLine = ref_file.readline()
        aLine = aLine.rstrip('\n')
        return aLine