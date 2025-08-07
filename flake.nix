{
  description = "SUNDIALS environment flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, flake-utils, nixpkgs }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python-with-packages = pkgs.python312.withPackages (ps: [
          ps.pytest
          ps.hypothesis
          ps.numpy
          ps.pandas
          ps.sympy
          ps.pydantic
          ps.black
          ps.mypy
          ps.ipython
          ps.jupyter
          ps.matplotlib
        ]);
      in {
        devShells.default = pkgs.mkShell {
          name = "SUNDIALS dev shell";
          buildInputs = pkgs.sundials.buildInputs ++ (with pkgs; [
            gdb
            clang-tools
            cmake-format
            bison
            flex
            nixfmt-classic
            python-with-packages
          ]);
          nativeBuildInputs = pkgs.sundials.nativeBuildInputs ++ [ ];
          shellHook = ''
            export KLU_INCLUDE_DIR="${pkgs.suitesparse.dev}/include"
            export KLU_LIBRARY_DIR="${pkgs.suitesparse}/lib"
          '';
        };
      });
}
