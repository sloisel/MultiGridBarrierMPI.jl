using Documenter
using MultiGridBarrierMPI
using Pkg

# Compute version dynamically
version = string(pkgversion(MultiGridBarrierMPI))

makedocs(;
    modules=[MultiGridBarrierMPI],
    authors="Sebastien Loisel and contributors",
    sitename="MultiGridBarrierMPI.jl $version",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sloisel.github.io/MultiGridBarrierMPI.jl",
        repolink="https://github.com/sloisel/MultiGridBarrierMPI.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Installation" => "installation.md",
        "User Guide" => "guide.md",
        "API Reference" => "api.md",
    ],
    repo=Documenter.Remotes.GitHub("sloisel", "MultiGridBarrierMPI.jl"),
    warnonly=true,  # Don't fail on warnings during development
)

deploydocs(;
    repo="github.com/sloisel/MultiGridBarrierMPI.jl",
    devbranch="main",
)
