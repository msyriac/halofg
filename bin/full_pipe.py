

cosmo_sims
cosmo_analysis


def parse_cosmology(Config,section):

    if sec_type=="pycamb_params":
        pass
    elif sec_type=="camb_file_root":
        pass
    elif sec_type=="enlib_file_root":
        pass


    return cc




cc_sims = parse_cosmology(Config,section_cosmo_sims)
cc_analysis = parse_cosmology(Config,section_cosmo_analysis) if section_cosmo_analysis!=section_cosmo_sims else cc_sims
