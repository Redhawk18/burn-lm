{
  description = "CUDA/ROCm env-shell";

  inputs = {
    nixpkgs.url = "github:cachix/devenv-nixpkgs/rolling";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
    devenv.inputs.nixpkgs.follows = "nixpkgs";

  };

  nixConfig = {
    extra-substituters = [
      "https://devenv.cachix.org"
      "https://cache.nixos-cuda.org"
    ];
    extra-trusted-public-keys = [
      "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw="
      "cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M="
    ];
  };

  outputs =
    {
      self,
      nixpkgs,
      devenv,
      systems,
      ...
    }@inputs:
    let
      forEachSystem = nixpkgs.lib.genAttrs (import systems);
    in
    {
      devShells = forEachSystem (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
            config.allowUnsupportedSystem = false; # By default we build broken packages.
          };
          lib = pkgs.lib;

          hasCuda = builtins.pathExists /dev/nvidiactl;
          hasRocm = builtins.pathExists /dev/kfd;

          cuda = pkgs.pkgs.cudaPackages_13_0;
          rocm = pkgs.rocmPackages;

          gpuPkgs =
            if hasCuda then
              [
                cuda.cudatoolkit
                cuda.cuda_nvcc
                cuda.libcublas
                cuda.cudnn
                cuda.nccl

              ]
            else if hasRocm then
              [
                rocm.clr
                rocm.rocblas
                rocm.rocm-smi
              ]
            else
              [ ];

          gpuHook =
            if hasCuda then
              # ""
              ''
                export CUDA_HOME=${cuda.cudatoolkit}
                export CUDA_PATH=${cuda.cudatoolkit}

                # On NixOs we need to look in `opengl-driver`.
                export LD_LIBRARY_PATH="/run/opengl-driver/lib:${cuda.nccl}/lib:${cuda.cudatoolkit}/lib:${pkgs.ncurses5}/lib:$LD_LIBRARY_PATH"
                export CPATH="${cuda.nccl}/include:${cuda.cudatoolkit}/include:$CPATH"
                export LIBRARY_PATH="${cuda.nccl}/lib:${cuda.cudatoolkit}/lib:$LIBRARY_PATH"

              ''
            else if hasRocm then
              ''
                export HIP_PATH=${rocm.clr}
                export ROCM_PATH=${rocm.clr}

                export EXTRA_LDFLAGS="-L${rocm.clr}/lib"
                export EXTRA_CCFLAGS="-I${rocm.clr}/include"
                export LD_LIBRARY_PATH="${rocm.clr}/lib:${pkgs.ncurses5}/lib:$LD_LIBRARY_PATH"

                # Helps if the device is either not supported or not found.
                # export HSA_OVERRIDE_GFX_VERSION=10.3.0 
              ''

            else
              "";

        in
        {
          default = devenv.lib.mkShell {
            inherit inputs pkgs;
            modules = [
              {
                languages.r.enable = true;
                languages.rust = {
                  enable = true;
                  # channel = "stable";
                  mold.enable = true;
                };

                # https://devenv.sh/reference/options/
                packages =
                  with pkgs;
                  [
                    just
                    openssl
                    rstudio
                    rsync
                    singularity

                  ]
                  ++ gpuPkgs;

                enterShell = ''
                  ${gpuHook}
                  echo Welcome! Please see the README.md for instructions.
                  echo
                '';

                env = {
                  LD_LIBRARY_PATH = lib.makeLibraryPath (
                    with pkgs;
                    [
                      libGL
                      vulkan-loader # requires VK_EXT_shader_64bit_indexing
                    ]
                  );
                  APPTAINER_TMPDIR = "/home/tmp/";
                  RUST_LOG = "info";
                  SINGULARITY_TMPDIR = "/home/tmp/";
                };

              }
            ];
          };
        }
      );
    };
}
