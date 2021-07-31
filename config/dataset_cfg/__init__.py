def get_config(name, args=None):
    if '+' in name:
        from .opt_mix_data import Config
    elif name == 'nr':
        from .opt_NR import Config
    elif name == 'aid':
        from .opt_AID import Config
    elif name == 'eurorgb':
        from .opt_EuroSAT_RGB import Config
    elif name == 'euroms':
        from .opt_EuroSAT_MS import Config
    elif name == 'euromsrgb':
        from .opt_EuroSAT_MSRGB import Config

    if args is not None:
        mOptions = Config(args)
    else:
        mOptions = Config()

    return mOptions

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR+'/Tools/dltoos')
