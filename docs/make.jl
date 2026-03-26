using Documenter
using GraphTransportation

# Keep examples/README.md as the single source of truth; copy it into
# docs/src/ so Documenter can include it as a page.
cp(joinpath(@__DIR__, "..", "examples", "README.md"),
   joinpath(@__DIR__, "src", "examples.md"); force=true)

makedocs(
    sitename = "GraphTransportation.jl",
    authors = "David Gentile, James M. Murphy",
    modules = [GraphTransportation],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://dcgentile.github.io/GraphTransportation.jl",
        edit_link = "main",
    ),
    pages = [
        "Home"      => "index.md",
        "Examples"  => "examples.md",
        "API"       => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/dcgentile/GraphTransportation.jl",
    devbranch = "main",
)
