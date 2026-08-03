import hydra
from omegaconf import OmegaConf

from run_naming import variant_tag

@hydra.main(config_path=None)
def register_resolvers(cfg):
    pass

# Define the resolver function
def replace_slash(value: str) -> str:
    return value.replace('/', '_')

def replace_substring(value: str, old: str, new: str) -> str:
    return str(value).replace(str(old), str(new))

# Register the resolver with Hydra
OmegaConf.register_new_resolver("replace_slash", replace_slash)
OmegaConf.register_new_resolver("replace_substring", replace_substring)
# '' for the original defaults -> pre-existing run dirs stay byte-identical.
OmegaConf.register_new_resolver("variant_tag", variant_tag)

if __name__ == "__main__":
    register_resolvers()
