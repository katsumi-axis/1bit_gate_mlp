{
  description = "MLX 1D-MNIST experiments with uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.python312
            pkgs.uv
            pkgs.git
          ];

          env = {
            UV_PYTHON = "${pkgs.python312}/bin/python";
            UV_PROJECT_ENVIRONMENT = ".venv";
          };

          shellHook = ''
            echo "Python: $(${pkgs.python312}/bin/python --version)"
            echo "uv: $(uv --version)"
          '';
        };
      }
    );
}
