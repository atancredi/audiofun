from mutagen import File as MutagenFile


def copy_metadata(input_path: str, output_path: str):
    """
    Copy metadata from 'input_path' to 'output_path'
    """
    try:
        input_meta = MutagenFile(input_path, easy=True)
        if input_meta is None:
            return
        output_meta = MutagenFile(output_path, easy=True)
        if output_meta is not None:
            for key in input_meta.keys():
                output_meta[key] = input_meta[key]
            output_meta.save()
    except Exception as e:
        print(f"Metadata copy failed: {e}")
