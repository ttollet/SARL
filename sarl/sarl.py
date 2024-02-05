import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="config", config_name="sarl")
def main(hydra_config: DictConfig):
    
    # Proof of concept: hydra for read/write experiments
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # See /.hydra in relevant folder for config
    from tempdir_do_not_use.test_pdqn import test_pdqn_platform
    return test_pdqn_platform(**hydra_config)


if __name__ == "__main__":
    main()