using Documenter
using HPCMultiGridBarrier
using Pkg

# Compute version dynamically
version = string(pkgversion(HPCMultiGridBarrier))

makedocs(;
    modules=[HPCMultiGridBarrier],
    authors="Sebastien Loisel and contributors",
    sitename="HPCMultiGridBarrier.jl $version",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sloisel.github.io/HPCMultiGridBarrier.jl",
        repolink="https://github.com/sloisel/HPCMultiGridBarrier.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Installation" => "installation.md",
        "User Guide" => "guide.md",
        "API Reference" => "api.md",
    ],
    repo=Documenter.Remotes.GitHub("sloisel", "HPCMultiGridBarrier.jl"),
    warnonly=true,  # Don't fail on warnings during development
)

deploydocs(;
    repo="github.com/sloisel/HPCMultiGridBarrier.jl",
    devbranch="main",
)
