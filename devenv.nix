{ pkgs, lib, config, inputs, ... }:
{
  name = "voice-pulse";

  env.PYTHONPATH = ".";

  # https://devenv.sh/packages/
  packages = [ pkgs.git pkgs.zsh pkgs.portaudio ];

  # https://devenv.sh/scripts/
  # scripts.hello.exec = "";

  enterShell = '''';

  # https://devenv.sh/tests/
  enterTest = '''';

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    version = "3.9";

    poetry = {
      enable = true;
      activate.enable = true;
      install = {
        enable = true;
        allExtras = true;
      };
    };
  };

  # https://devenv.sh/processes/
  # processes.my_service.exec = "";
}
