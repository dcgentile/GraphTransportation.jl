using Documenter
using GraphTransportation

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
        "Home"    => "index.md",
        "API"     => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/dcgentile/GraphTransportation.jl",
    devbranch = "main",
)
